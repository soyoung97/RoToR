cd ..
GPU=5
MODEL_G=4B
METHOD=ours
SORT=lexical

CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data mintaka --method $METHOD --split 30
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data mintaka --method $METHOD --split 30 --mode random_shuffle --seed 0
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data mintaka --method $METHOD --split 30 --mode random_shuffle --seed 1
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data mintaka --method $METHOD --split 30 --mode random_shuffle --seed 2


CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data mintaka --method $METHOD --split 50
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data mintaka --method $METHOD --split 50 --mode random_shuffle --seed 0
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data mintaka --method $METHOD --split 50 --mode random_shuffle --seed 1
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data mintaka --method $METHOD --split 50 --mode random_shuffle --seed 2


CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data webqsp --method $METHOD --split 30
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data webqsp --method $METHOD --split 30 --mode random_shuffle --seed 0
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data webqsp --method $METHOD --split 30 --mode random_shuffle --seed 1
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data webqsp --method $METHOD --split 30 --mode random_shuffle --seed 2


CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data webqsp --method $METHOD --split 50
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data webqsp --method $METHOD --split 50 --mode random_shuffle --seed 0
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data webqsp --method $METHOD --split 50 --mode random_shuffle --seed 1
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL_G-Chat --name 0210 --sorting_method $SORT --data webqsp --method $METHOD --split 50 --mode random_shuffle --seed 2




