cd ..
for mode in {0..23}
do
    echo "Running with mode=$mode"
    CUDA_VISIBLE_DEVICES=3 python3 run.py \
        --name perm \
        --data mmlu \
        --model_name Qwen/Qwen1.5-4B-Chat \
        --split mmlu_full \
        --method pine \
        --inference_type log_likelihood \
        --mode $mode

done
for mode in {0..23}
do
    echo "Running with mode=$mode"
    CUDA_VISIBLE_DEVICES=3 python3 run.py \
        --name perm \
        --data mmlu \
        --model_name Qwen/Qwen1.5-4B-Chat \
        --split mmlu_full \
        --method ours \
        --inference_type log_likelihood \
        --mode $mode

done
for mode in {0..23}
do
    echo "Running with mode=$mode"
    CUDA_VISIBLE_DEVICES=3 python3 run.py \
        --name perm \
        --data mmlu \
        --model_name Qwen/Qwen1.5-4B-Chat \
        --split mmlu_full \
        --method ours \
        --sorting_method monot5 \
        --inference_type log_likelihood \
        --mode $mode

done


for mode in {0..23}
do
    echo "Running with mode=$mode"
    CUDA_VISIBLE_DEVICES=3 python3 run.py \
        --name perm \
        --data mmlu \
        --model_name Qwen/Qwen1.5-4B-Chat \
        --split mmlu_full \
        --method ours \
        --sorting_method freq \
        --inference_type log_likelihood \
        --mode $mode

done
