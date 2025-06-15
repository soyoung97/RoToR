GPUS="7"
FILENAME="nq-open-10_total_documents_gold_at_4.json"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

python eval/lost_in_middle_eval.py  \
    --eval_data lost_in_middle \
    --filename ${FILENAME} \
    --eval_num 100 \
    --gpu ${GPUS} \
    --mode origin \
    --model_name ${MODEL_NAME} \
