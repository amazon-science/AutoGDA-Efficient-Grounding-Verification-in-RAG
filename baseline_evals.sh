#!/bin/bash
for groupname in sales_call ectsum scitldr shakespeare billsum; do #
#for groupname in lifestyle recreation tech; do # news qmsumm ectsum scitldr shakespeare sales_email billsum
#export PYTHONPATH="."; python3 src/llm_sync_data/generate_summedits_sync.py -d lfrqa-${groupname} --n_generate 4
export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=2,3; python3 src/sync_data/optimization_initial_script.py -e 20 --num_iters 4 -m bart-large tasksource tasksource_v1 --run c_opt20_${groupname} --dataset summedits-${groupname}
done
