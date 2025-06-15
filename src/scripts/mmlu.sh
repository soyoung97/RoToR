#!/bin/bash

# Read command-line arguments
GPU=$1
MODEL_SIZE=$2  # First argument, e.g., "4B"
MODE=$3 # orig, ours, pine
SPLIT_TYPE=$4  # Second argument, e.g., "freq"

cd ..
# Loop over modes from 0 to 23
for mode in {0..23}
do
    echo "Running with GPU=$GPU, mode=$mode, model_size=$MODEL_SIZE, method=$METHOD, sort_type=$SPLIT_TYPE"

    CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --name arrfeb \
        --data mmlu \
        --model_name Qwen/Qwen1.5-$MODEL_SIZE-Chat \
        --method ${MODE} \
        --sorting_method ${SPLIT_TYPE} \
        --inference_type log_likelihood \
        --mode $mode
done
