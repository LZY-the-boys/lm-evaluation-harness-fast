import transformers
import torch

# 0. 5min 磁盘io
# model = transformers.LlamaForCausalLM.from_pretrained(
#     '/data/outs/llama2-13b-sharegpt4-orca-platypus/ep_4',
#     torch_dtype=torch.bfloat16,
# )

# 1. 1min 速度明显快了
# model = transformers.LlamaForCausalLM.from_pretrained(
#     '/data/outs/llama2-13b-sharegpt4-orca-platypus/ep_4',
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     # The model is first created on the Meta device (with empty weights)
#     # the state dict is then loaded inside it (shard by shard in the case of a sharded checkpoint). 
#     # This way the maximum RAM used is the full size of the model only.
# )

# 2. 也是大概1min
# model = transformers.LlamaForCausalLM.from_pretrained(
#     '/data/outs/llama2-13b-sharegpt4-orca-platypus/ep_4',
#     torch_dtype=torch.bfloat16,
#     device_map='auto',
#     # With device_map, low_cpu_mem_usage is automatically set to True
#     # With device_map="auto", Accelerate will determine where to put each layer to maximize the use of your fastest devices (GPUs) and offload the rest on the CPU
# )

# 3. 速度貌似最快
# model = transformers.LlamaForCausalLM.from_pretrained(
#     '/data/outs/llama2-13b-sharegpt4-orca-platypus/ep_4',
#     device_map={'': 0},
#     torch_dtype=torch.bfloat16,
# )

# 4.
# model = transformers.LlamaForCausalLM.from_pretrained(
#     '/data/outs/llama2-13b-sharegpt4-orca-platypus/ep_4',
#     device_map={'': 'cuda:0'},
#     torch_dtype=torch.bfloat16,
# )
# model = transformers.LlamaForCausalLM.from_pretrained(
#     '/data/outs/llama2-13b-sharegpt4-orca-platypus/ep_4',
#     device_map={'': 'cuda'},
#     torch_dtype=torch.bfloat16,
# )


# 5. Loading GPTQ quantized model requires optimum library : `pip install optimum` and auto-gptq library 'pip install auto-gptq'
# model = transformers.AutoModelForCausalLM.from_pretrained(
#     '/data/models/Llama-2-70B-GPTQ',
#     # device_map={'': 'cuda'},
#     device_map='auto',
#     torch_dtype=torch.float16,
# )
# import pdb; pdb.set_trace()

