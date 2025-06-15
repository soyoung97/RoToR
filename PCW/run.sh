CUDA_VISIBLE_DEVICES=6 python run_evaluation.py \
--dataset sst2 \
--model meta-llama/Meta-Llama-3-8B \
--n-windows 1 \
--n-windows 3 \
--subsample-test-set 250 \
--n-runs 30 \
--output-dir ./outputs/
