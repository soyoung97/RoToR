cd ..
for mode in {0..23}
do
    echo "Running with mode=$mode"
    CUDA_VISIBLE_DEVICES=7 python3 run.py \
        --name perm \
        --data mmlu \
        --model_name Qwen/Qwen1.5-7B-Chat \
        --split mmlu_full \
        --method ours \
        --sorting_method freq \
        --inference_type log_likelihood \
        --mode $mode

done
