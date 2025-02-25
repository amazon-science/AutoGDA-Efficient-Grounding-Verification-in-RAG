#!/bin/bash
# call args: device_no study dataset group init_model target_model
for seed in 1 2 3 4 5; do
export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/reevaluate_test.py -f eval_run-$2/seed_${seed}-$3 -d $4 -g $5 --model tasksource
done