import torch
import transformers
from typing import Optional, Union
from lm_eval.base import BaseLM
from accelerate import Accelerator, find_executable_batch_size, DistributedType
import multiprocess
from tqdm import tqdm
from transformers import GPTQConfig,BitsAndBytesConfig
from peft import PeftModelForCausalLM

self_tokenizer=None
self_eot_token_id=None
def _tokenize_fn(request):

    def self_tok_encode(string):
        return self_tokenizer.encode(string, add_special_tokens=False)

    def self_encode_pair(context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self_tok_encode(context + continuation)
        context_enc = self_tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    context, continuation = request
    if context == "":
        # end of text as context
        context_enc, continuation_enc = [self_eot_token_id], self_tok_encode(continuation)
    else:
        context_enc, continuation_enc = self_encode_pair(context, continuation)
    
    return ((context, continuation), context_enc, continuation_enc)

def _get_dtype(
    dtype: Union[str, torch.dtype]
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class HFLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_length=None,
        # load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]]="auto",
        tensor_parallel=False, # tensor parallel
        peft: Optional[str] = None,
        quantization_config = None,
        flash_attention = False,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator = Accelerator()
        model_kwargs = {}

        self.rank = 0
        self.world_size = 1
        data_parallel = False
        self._device = torch.device(f"cuda:0")
        if not (tensor_parallel or accelerator.num_processes > 1):
            # single gpu
            model_kwargs = {'device_map': {'':0}}
        elif gpus > 1:
            assert not (accelerator.num_processes > 1 and tensor_parallel), (
                    "Attempted to use both a HF Accelerate `device_map` and to launch via `accelerate launch`. If this is the case, please either remove `parallelize=True` from --model_args or launch outside of the Accelerate launcher."
                )

            # multi gpu
            if tensor_parallel:
                # tensor parallel
                model_kwargs = {'device_map':'auto'}
            else:
                assert not gpus > accelerator.num_processes, (
                    "set CUDA_VISIBLE_DEVICES; temporially not support gpus > num_processes"
                )
                data_parallel = True
                model_kwargs = {'device_map': {'':accelerator.local_process_index}}

                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self.accelerator = accelerator
                self.rank = self.accelerator.local_process_index
                self.world_size = self.accelerator.num_processes

        revision = revision + ("/" + subfolder if subfolder is not None else "")
        # fix tokenize speed:
        config = transformers.AutoConfig.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
        )
        if isinstance(config, transformers.LlamaConfig):
            # transformers.AutoModelForCausalLM = transformers.LlamaForCausalLM
            transformers.AutoTokenizer = transformers.LlamaTokenizer

            if flash_attention:
                from lm_eval.models.flash_attn_patch import replace_llama_attn_with_flash_attn
                replace_llama_attn_with_flash_attn(packed=False)
                if self.rank == 0:
                    print(f'>>> use flash_attention')

        if quantization_config:
            # NOTICE: model.config.quantization_config > input quantization_config
            if quantization_config['quant_method'] == 'gptq':
                quantization_config = GPTQConfig.from_dict(quantization_config)
            elif quantization_config['quant_method'] == 'bitsandbytes':
                quantization_config['bnb_4bit_compute_dtype'] = _get_dtype(dtype)
                if self.rank == 0:
                    print(f'>>> set bnb_4bit_compute_dtype to {_get_dtype(dtype)}')
                quantization_config = BitsAndBytesConfig.from_dict(quantization_config)  
            else:
                raise Exception('wrong quantization_config')      
            model_kwargs.update({'quantization_config': quantization_config})

        # support for auto_gptq
        torch_dtype=_get_dtype(dtype)
        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            low_cpu_mem_usage=True, # loading speedup 
            revision=revision,
            torch_dtype=_get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        ).eval()
        if self.rank == 0:
            print(self.gpt2.config)

        if peft:
            
            if flash_attention:
                for name, module in self.gpt2.named_modules():
                    if "norm" in name:
                        module.to(torch_dtype)
                    if "lm_head" in name or "embed_tokens" in name:
                        if hasattr(module, "weight"):
                            module.to(torch_dtype)
            
            self.gpt2 = PeftModelForCausalLM.from_pretrained(
                self.gpt2, peft
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.vocab_size = self.tokenizer.vocab_size

        # setup for automatic batch size detection
        if batch_size == "auto":
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size)

        self._max_length = max_length

        if data_parallel:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self.gpt2 = accelerator.prepare(self.gpt2)
            else:
                self.gpt2 = accelerator.prepare_model(self.gpt2, evaluation_mode=True)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length: # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.gpt2.config, attr):
                return getattr(self.gpt2.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH


    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.gpt2.generate(context, **generation_kwargs)

    # multiple process 注意这里是需要保持顺序的
    def loglikelihood(self, requests):

        global self_eot_token_id, self_tokenizer
        self_eot_token_id, self_tokenizer = self.eot_token_id, self.tokenizer
        
        process_num = 20
        # default multiprocessing start method in Python is "fork," 
        # which clones the current process, including its CUDA context, lead to error
        # global eot_token_id, tokenizer, add_special_tokens
        # eot_token_id, tokenizer, add_special_tokens = self.eot_token_id, self.tokenizer, self.add_special_tokens
        with multiprocess.Pool(process_num) as pool:
            new_reqs = list(tqdm(
                # chunksize=100 
                # pool.imap_unordered(_tokenize_fn, requests),  # 不保持顺序
                pool.imap(_tokenize_fn, requests), # 保持顺序
                total=len(requests), 
                desc=f'Tokenize MAP({process_num})'
            ))
        return self._loglikelihood_tokens(new_reqs)

# for backwards compatibility
GPT2LM = HFLM
