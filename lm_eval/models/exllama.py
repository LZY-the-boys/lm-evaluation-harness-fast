import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from lm_eval.base import BaseLM
import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config,ExLlamaV2Tokenizer
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import Accelerator, find_executable_batch_size, DistributedType
import multiprocess
from tqdm import tqdm

self_tokenizer=None
self_eot_token_id=None
def _tokenize_fn(request):

    def self_tok_encode(string):
        # will not add default eos
        # will be [[]] format
        # default return a tensor
        return self_tokenizer.encode(string, add_bos=True)[0].tolist()

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

class Exllamav2ModelHF(PreTrainedModel):
    def __init__(
        self, 
        config: ExLlamaV2Config, 
        gpu_split=None, # tensor_parallel & data_parallel
        batch_size=1,
        cfg_cache=True,
    ):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlamaV2(config)
        split = None
        if gpu_split:
            split = [float(alloc) for alloc in gpu_split.split(";")]
        self.ex_model.load(split)

        self.generation_config = GenerationConfig()

        self.ex_cache = ExLlamaV2Cache(self.ex_model, batch_size)
        self.past_seq = None
        self.cfg_cache = cfg_cache
        if cfg_cache:
            self.ex_cache_negative = ExLlamaV2Cache(self.ex_model)
            self.past_seq_negative = None

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return self.device

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)
        past_key_values = kwargs.get('past_key_values', None)

        if len(args) > 0:
            if not self.cfg_cache:
                raise Exception("Please enable the cfg-cache option to use CFG with ExLlamav2_HF.")

            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            ex_cache = self.ex_cache_negative
        else:
            input_ids = kwargs['input_ids']
            is_negative = False
            past_seq = self.past_seq
            ex_cache = self.ex_cache

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)

        # Make the forward call
        if labels is None:
            # RNN式写法，最终返回1个token的logit
            if past_seq is None or not torch.equal(past_seq, seq_tensor[:-1]):
                ex_cache.current_seq_len = 0
                self.ex_model.forward(torch.tensor([seq[:-1]], dtype=torch.long), ex_cache, preprocess_only=True)

            logits = self.ex_model.forward(torch.tensor([seq[-1:]], dtype=torch.long), ex_cache).to(input_ids.device)
        else:
            ex_cache.current_seq_len = 0
            # logits = self.ex_model.forward(torch.tensor([seq], dtype=torch.long), ex_cache, last_id_only=False)
            logits = self.ex_model.forward(torch.tensor([seq], dtype=torch.long), ex_cache)

        if is_negative:
            self.past_seq_negative = seq_tensor
        else:
            self.past_seq = seq_tensor

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(logits=logits, past_key_values=seq if use_cache else None, loss=loss)

    def simple_forward(self, inputs):
        # reset cache
        self.ex_cache.current_seq_len=0
        return self.ex_model.forward(inputs, self.ex_cache)

class ExllamaLM(BaseLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained="LLama2-70B-chat-2.55bpw-h6-exl2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_length=None,
        dtype: Optional[Union[str, torch.dtype]]="auto",
        gpu_split=None, # tensor_parallel
        peft: Optional[str] = None,
        # quantization_config = None,
    ):
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        batch_size = int(batch_size)

        gpus = torch.cuda.device_count()
        accelerator = Accelerator()
        self.rank = 0
        self.world_size = 1
        data_parallel = False
        self._device = torch.device(f"cuda:0")

        if gpus > 1 and not gpu_split:
            data_parallel = True
            gpu_split = ['0']*gpus
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.accelerator = accelerator
            self.rank = self.accelerator.local_process_index
            self.world_size = self.accelerator.num_processes
            gpu_split[self.rank] = '40'
            gpu_split=';'.join(gpu_split)
            # still exists bug: 

        config = ExLlamaV2Config()
        config.model_dir = str(pretrained)
        config.prepare()
        # config.max_seq_len = kwargs['args.max_seq_len']
        # config.scale_pos_emb = kwargs['compress_pos_emb']
        # config.scale_alpha_value = kwargs['alpha_value']
        self._max_length = max_length

        # NOTE: device is controlled by gpu_split
        self.model = Exllamav2ModelHF(
            config,
            gpu_split=gpu_split,
            batch_size=batch_size,
        ).eval()

        self.tokenizer = ExLlamaV2Tokenizer(config)

        # setup for automatic batch size detection
        if batch_size == "auto":
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size)

        if data_parallel:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self.model = accelerator.prepare(self.model)
            else:
                self.model = accelerator.prepare_model(self.model, evaluation_mode=True)


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length: # if max length manually set, return it
            return self._max_length
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
        with torch.inference_mode():
            # return self.model(inps)
            logits = self.model.simple_forward(inps)
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)

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