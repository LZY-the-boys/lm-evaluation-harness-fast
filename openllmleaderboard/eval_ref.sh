# 检查benchmark是否能够复现结果
export MODEL_DIR=openchat/openchat_v3.2
export OUR_DIR=outs/ref
export XDG_CACHE_HOME=/data/.cache

# single GPU
# export CUDA_VISIBLE_DEVICES=1

# python main.py \
#     --model hf-causal \
#     --model_args pretrained=$MODEL_DIR,dtype="bfloat16" \
#     --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics \
#     --num_fewshot=5 \
#     --no_cache \
#     --output_path $OUR_DIR/ref/mmlu \
#     --batch_size=1

# multi GPU
export CUDA_VISIBLE_DEVICES=2,3,4,5

# accelerate launch main.py \
#     --model hf-causal \
#     --model_args pretrained=$MODEL_DIR,dtype="bfloat16" \
#     --tasks arc_challenge \
#     --num_fewshot=25 \
#     --output_path $OUR_DIR/ref/arc \
#     --no_cache \
#     --batch_size=4 

# accelerate launch main.py \
#     --model hf-causal \
#     --model_args pretrained=$MODEL_DIR,dtype="bfloat16" \
#     --tasks hellaswag \
#     --num_fewshot=10 \
#     --output_path $OUR_DIR/ref/hellaswag \
#     --no_cache \
#     --batch_size=8 

# accelerate launch main.py \
#     --model hf-causal \
#     --model_args pretrained=$MODEL_DIR,dtype="bfloat16" \
#     --tasks truthfulqa_mc \
#     --output_path $OUR_DIR/ref/truthfulqa \
#     --num_fewshot=0 \
#     --no_cache \
#     --batch_size=24

accelerate launch main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_DIR,dtype="bfloat16" \
    --tasks  hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics  \
    --num_fewshot=5 \
    --no_cache \
    --output_path $OUR_DIR/ref/mmlu \
    --batch_size=1