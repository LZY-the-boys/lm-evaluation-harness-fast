# lm-evaluation-harness Speedup

The `lm-evaluation-harness` is very good and adopted in hotest leaderboard: [huggingface openllmleaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), 
but the [version](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463) huggingface used is very slow in terms of loading model,tokenize prompt and inference only on a single card.
In my test, the lastest lm-evaluation-harness version (`bigrefactor` branch) cannot reproduce huggingface openllmleaderboard result.

So I add speedup teniques totally based on [this version](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463).

1. add `low_cpu_mem_usage` / `device_map`, which can speedup model loading vastly.
2. convert `autotokenizer` to `llamatokenizer`, which can surely improve tokenize speed.
3. add multiprocess tokenizer.
4. add multigpu accelerater inference:
   1. `tensor parallel`: 1 process, split model across multiple gpu.
   2. `data parallel inference`: multiple process, split data across multiple gpu.

I also add support for auto-gptq, bitsandbytes, peft and exllamav2 in lm-evaluation-harness !
For example, The 70B [`LLama2-70B-chat-2.55bpw-h6-exl2`](https://huggingface.co/turboderp/LLama2-70B-chat-2.55bpw-h6-exl2) only need 24~25 VRAM to load. 

The Usage for ordinary huggingface models: 
```
# if single gpu
python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_DIR,dtype="bfloat16",trust_remote_code=True \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics \
    --num_fewshot=5 \
    --no_cache \
    --output_path $OUR_DIR/ref/mmlu \
    --batch_size=2

# if multiple gpu tensor_parallel
python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_DIR,dtype="bfloat16",trust_remote_code=True,tensor_parallel=True \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics \
    --num_fewshot=5 \
    --no_cache \
    --output_path $OUR_DIR/ref/mmlu \
    --batch_size=2

# if multiple gpu data_parallel
accelerate launch main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_DIR,dtype="bfloat16",trust_remote_code=True \
    --tasks arc_challenge \
    --num_fewshot=25 \
    --output_path $OUR_DIR/ref/arc \
    --no_cache \
    --batch_size=4 
```

The Usage for huggingface gptq models/bitsandbytes models, only have to set `quantization_config`: 
```
accelerate launch main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_DIR,dtype="float16",max_length=1024 \
    --quantization_config="{\"bits\": 4, \"disable_exllama\":false,\"quant_method\":\"gptq\",\"use_cuda_fp16\":false}" \
    --tasks arc_challenge \
    --num_fewshot=25 \
    --output_path $OUR_DIR/ref/arc \
    --no_cache \
    --batch_size=1

accelerate launch main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_DIR,dtype="bfloat16" \
    --quantization_config="{\"load_in_8bit\":false,\"load_in_4bit\":true,\"bnb_4bit_use_double_quant\":true,\"bnb_4bit_quant_type\":\"nf4\",\"llm_int8_has_fp16_weight\":true,\"quant_method\":\"bitsandbytes\"}" \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics \
    --num_fewshot=5 \
    --no_cache \
    --output_path $OUR_DIR/ref/mmlu \
    --batch_size=1
```

The usage for exllamav2:
```
# for tensor_parallel: set the gpu_split, e.g. gpu_split="12;13" means load 12G of model split in gpu0 and 13G in gpu1.
accelerate launch main.py \
    --model exllama2 \
    --model_args pretrained=$MODEL_DIR,dtype="bfloat16",gpu_split="12;13" \
    --tasks arc_challenge \
    --num_fewshot=25 \
    --output_path $OUR_DIR/ref/arc \
    --no_cache \
    --batch_size=16

# for data_parallel
accelerate launch main.py \
    --model exllama2 \
    --model_args pretrained=$MODEL_DIR,dtype="bfloat16" \
    --tasks arc_challenge \
    --num_fewshot=25 \
    --output_path $OUR_DIR/ref/arc \
    --no_cache \
    --batch_size=16
```

For model like `OpenChat-v3.2`, we have results as follows and is the same as result of [this version](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463) on my machines, which is also very close to huggingface result.

```json
{
  "results": {
    "hendrycksTest-abstract_algebra": {
      "acc": 0.33,
      "acc_stderr": 0.04725815626252606,
      "acc_norm": 0.33,
      "acc_norm_stderr": 0.04725815626252606
    },
    "hendrycksTest-anatomy": {
      "acc": 0.4666666666666667,
      "acc_stderr": 0.043097329010363554,
      "acc_norm": 0.4666666666666667,
      "acc_norm_stderr": 0.043097329010363554
    },
    "hendrycksTest-astronomy": {
      "acc": 0.5657894736842105,
      "acc_stderr": 0.04033565667848319,
      "acc_norm": 0.5657894736842105,
      "acc_norm_stderr": 0.04033565667848319
    },
    "hendrycksTest-business_ethics": {
      "acc": 0.54,
      "acc_stderr": 0.05009082659620332,
      "acc_norm": 0.54,
      "acc_norm_stderr": 0.05009082659620332
    }
  },
  "versions": {
    "hendrycksTest-abstract_algebra": 1,
    "hendrycksTest-anatomy": 1,
    "hendrycksTest-astronomy": 1,
    "hendrycksTest-business_ethics": 1
  },
  "config": {
    "model": "hf-causal",
    "model_args": "pretrained=openchat/openchat_v3.2,dtype=bfloat16",
    "num_fewshot": 5,
    "batch_size": "1",
    "batch_sizes": [],
    "device": null,
    "no_cache": true,
    "limit": null,
    "bootstrap_iters": 100000,
    "description_dict": {}
  }
}
```

# Language Model Evaluation Harness

## Notice to Users
(as of 6/15/23)
We have a revamp of the Evaluation Harness library internals staged on the [big-refactor](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) branch! It is far along in progress, but before we start to move the `master` branch of the repository over to this new design with a new version release, we'd like to ensure that it's been tested by outside users and there are no glaring bugs.

We’d like your help to test it out! you can help by:
1. Trying out your current workloads on the big-refactor branch, and seeing if anything breaks or is counterintuitive,
2. Porting tasks supported in the previous version of the harness to the new YAML configuration format. Please check out our [task implementation guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/docs/new_task_guide.md) for more information.

If you choose to port a task not yet completed according to [our checklist](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/tasks/README.md), then you can contribute it by opening a PR containing [Refactor] in the name with: 
- A shell command to run the task in the `master` branch, and what the score is
- A shell command to run the task in your PR branch to `big-refactor`, and what the resulting score is, to show that we achieve equality between the two implementations.

Lastly, we'll no longer be accepting new feature requests beyond those that are already open to the master branch as we carry out this switch to the new version over the next week, though we will be accepting bugfixes to `master` branch and PRs to `big-refactor`. Feel free to reach out in the #lm-thunderdome channel of the EAI discord for more information.


## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for commercial APIs including [OpenAI](https://openai.com), [goose.ai](https://goose.ai), and [TextSynth](https://textsynth.com/).
- Support for evaluation on adapters (e.g. LoRa) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Evaluating with publicly available prompts ensures reproducibility and comparability between papers.
- Task versioning to ensure reproducibility when tasks are updated.

## Install

To install `lm-eval` from the github repository main branch, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages, you must install the package with the `multilingual` extra:

```bash
pip install -e ".[multilingual]"
```

To support loading GPTQ quantized models, install the package with the `auto-gptq` extra:

```bash
pip install -e ".[auto-gptq]"
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command:


```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints, or to specify the datatype for running a model:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0
```

To evaluate models that are loaded via `AutoSeq2SeqLM` in Huggingface, you instead use `hf-seq2seq`. *To evaluate (causal) models across multiple GPUs, use `--model hf-causal-experimental`*

> **Warning**: Choosing the wrong model may result in erroneous outputs despite not erroring.

### Commercial APIs

Our library also supports language models served via the OpenAI API:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially maintained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library. These include [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md), and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.

## Advanced Usage

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=EleutherAI/gpt-j-6b,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

GPTQ quantized models can be loaded by specifying their file names in `,quantized=NAME` (or `,quantized=True` for default names) in the `model_args` argument:

```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=model-name-or-path,quantized=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

We currently only support one prompt per task, which we strive to make the "standard" as defined by the benchmark's authors. If you would like to study how varying prompts causes changes in the evaluation score, check out the [BigScience fork](https://github.com/bigscience-workshop/lm-evaluation-harness) of this repo. We are currently working on upstreaming this capability to `main`.

## Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/task_guide.md).

## Task Versioning

To help improve reproducibility, all tasks have a `VERSION` field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. The purpose of the version is so that if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. To enforce this, there are unit tests that make sure the behavior of all tests remains the same as when they were first implemented. Task versions start at 0, and each time a breaking change is made, the version is incremented by one.

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

## Test Set Decontamination

To address concerns about train / test contamination, we provide utilities for comparing results on a benchmark using only the data points not found in the model training set. Unfortunately, outside of models trained on the Pile and C4, its very rare that people who train models disclose the contents of the training data. However this utility can be useful to evaluate models you have trained on private data, provided you are willing to pre-compute the necessary indices. We provide computed indices for 13-gram exact match deduplication against the Pile, and plan to add additional precomputed dataset indices in the future (including C4 and min-hash LSH deduplication).

For details on text decontamination, see the [decontamination guide](./docs/decontamination.md).

Note that the directory provided to the `--decontamination_ngrams_path` argument should contain the ngram files and info.json. See the above guide for ngram generation for the pile, this could be adapted for other training sets.

```bash
python main.py \
    --model gpt2 \
    --tasks sciq \
    --decontamination_ngrams_path path/containing/training/set/ngrams \
    --device cuda:0
```

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
