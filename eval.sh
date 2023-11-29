source activate lla
# set -x
if [ ! -d $home ]; then
    echo "home is not set"
    exit 1
fi
if [ ! -d $peft ]; then
    echo "peft is not set"
    exit 1
fi
if [ -z "$out_dir" ]; then
    echo "out_dir is not set"
    exit 1
fi

cd $LZY_HOME/lm-evaluation-harness-leaderboard

function eval_single() {
# args: peft task
if [[ $peft == '7b' ]]; then
    model_args="pretrained=$LLAMA7B,dtype=bfloat16"
elif [[ $peft == '13b' ]]; then
    model_args="pretrained=$LLAMA13B,dtype=bfloat16"
elif [[ $peft == '70b' ]]; then
    model_args="pretrained=$LLAMA70B,dtype=bfloat16"
else
    if [ ! -d "$peft" ]; then 
        echo "Directory $peft does not exist."
        exit 1  # Exit the script with an error code
    fi

    name="$(basename $peft)"

    if [[ $peft == *7b* ]]; then
        model="$LLAMA7B"
    elif [[ $peft == *13b* ]]; then
        model="$LLAMA13B"
    elif [[ $peft == *70b* ]]; then
        model="$LLAMA70B"
    else
        echo "cannot decide pretrained"
        exit 1
    fi
    model_args="pretrained=$model,dtype=bfloat16,peft=$peft"
fi

out_name=$out_dir/$task 

if [[ $task == *truthfulqa* ]];then
    fewshot=0
    batch=24
elif [[ $task == *arc* ]];then
    fewshot=25
    batch=6
    if [ "$HostType" == 'A100-80' ]; then
        batch=16
    fi
elif [[ $task == *hellaswag* ]];then
    fewshot=10
    batch=8
    if [ "$HostType" == 'A100-80' ]; then
        batch=16
    fi
elif [[ $task == *mmlu* ]];then
    fewshot=5
    batch=2
    task=hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
accelerate launch --main_process_port $(shuf -i25000-30000 -n1) main.py \
    --model hf-causal \
    --model_args $model_args \
    --quantization_config="{\"load_in_8bit\":false,\"load_in_4bit\":true,\"bnb_4bit_use_double_quant\":true,\"bnb_4bit_quant_type\":\"nf4\",\"llm_int8_has_fp16_weight\":true,\"quant_method\":\"bitsandbytes\"}" \
    --tasks $task \
    --output_path $out_name \
    --num_fewshot=$fewshot \
    --no_cache \
    --batch_size=$batch

}

function eval() {
: ${task:='all'}
if [[ $task == 'all' ]];then
    for task in truthfulqa_mc arc_challenge hellaswag mmlu ; do
        task=$task eval_single
    done
else
    eval_single
fi
}

eval  || sleep 20000