export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$(pwd)
dataset_name="yilunzhao/robut"
split_name="wtq"


# model_name='baffo32/decapoda-research-llama-7B-hf'
model_name='gpt2-large'
python pcw_on_robut.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --split_name ${split_name} \
  --device cuda
