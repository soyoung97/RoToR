cd ..
GPU=1
METHOD=ours
MODEL=4B
SORT=monot5

CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data mintaka --sorting_method $SORT --method $METHOD --split 30
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data mintaka --sorting_method $SORT --method $METHOD --split 30 --mode random_shuffle --seed 0
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data mintaka --sorting_method $SORT --method $METHOD --split 30 --mode random_shuffle --seed 1
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data mintaka --sorting_method $SORT --method $METHOD --split 30 --mode random_shuffle --seed 2


CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data mintaka --sorting_method $SORT --method $METHOD --split 50
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data mintaka --sorting_method $SORT --method $METHOD --split 50 --mode random_shuffle --seed 0
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data mintaka --sorting_method $SORT --method $METHOD --split 50 --mode random_shuffle --seed 1
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data mintaka --sorting_method $SORT --method $METHOD --split 50 --mode random_shuffle --seed 2


CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data webqsp --sorting_method $SORT --method $METHOD --split 30
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data webqsp --sorting_method $SORT --method $METHOD --split 30 --mode random_shuffle --seed 0
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data webqsp --sorting_method $SORT --method $METHOD --split 30 --mode random_shuffle --seed 1
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data webqsp --sorting_method $SORT --method $METHOD --split 30 --mode random_shuffle --seed 2


CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data webqsp --sorting_method $SORT --method $METHOD --split 50
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data webqsp --sorting_method $SORT --method $METHOD --split 50 --mode random_shuffle --seed 0
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data webqsp --sorting_method $SORT --method $METHOD --split 50 --mode random_shuffle --seed 1
CUDA_VISIBLE_DEVICES=$GPU python3 run.py  --model_name Qwen/Qwen1.5-$MODEL-Chat --onlyeval --name 0210 --data webqsp --sorting_method $SORT --method $METHOD --split 50 --mode random_shuffle --seed 2




