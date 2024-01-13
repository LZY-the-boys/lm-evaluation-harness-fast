source activate lla
# set -x
if [ ! -d $home ]; then
    echo "home is not set"
    exit 1
fi
if [ -z "$out_dir" ]; then
    echo "out_dir is not set"
    exit 1
fi

cd $LZY_HOME/lm-evaluation-harness-leaderboard

function eval_single() {

if [ -z $model ];then
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
else
    model_args="pretrained=$model,dtype=float16"  
    if [ ! -z $peft ]; then
        model_args="$model_args,peft=$peft"
    fi
fi
echo $model_args
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
elif [[ $task == *bigbench* ]];then
    fewshot=5
    batch=2
    task=bigbench_salient_translation_error_detection,bigbench_causal_judgement,bigbench_reasoning_about_colored_objects,bigbench_date_understanding,bigbench_formal_fallacies_syllogisms_negation,bigbench_dyck_languages,bigbench_navigate,bigbench_temporal_sequences,bigbench_logical_deduction_three_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_movie_recommendation,bigbench_logical_deduction_five_objects,bigbench_tracking_shuffled_objects_three_objects,bigbench_logical_deduction_seven_objects,bigbench_snarks,bigbench_geometric_shapes,bigbench_tracking_shuffled_objects_five_objects,bigbench_disambiguation_qa,bigbench_sports_understanding,bigbench_hyperbaton,bigbench_ruin_names
elif [[ $task == *winogrande* ]];then
    fewshot=5
    batch=2
    task=winogrande
elif [[ $task == *drop* ]];then
    fewshot=3
    batch=2
    task=drop  
elif [[ $task == *gsm8k* ]];then
    fewshot=5
    batch=2
    task=gsm8k  
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
    for task in mmlu truthfulqa_mc arc_challenge hellaswag winogrande gsm8k; do
        if [ ! -f $out_dir/$task.json ];then
            task=$task eval_single
        else
            echo ">>> skip $task"
        fi
    done
else
    eval_single
fi
}

eval  || sleep 20000