#!/bin/bash
# call args: device_no study dataset group init_model target_model
seed=$2
for mode in tasksource,noise=0.5 none; do
export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/reevaluate_test.py -f runs/rmiss_eval3_v${seed}_${mode}_ragtruth_QA -d ragtruth -g QA --model tasksource
done