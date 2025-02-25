#!/bin/bash
# call args: device_no study dataset group init_model target_model
for seed in 1 2 3 4 5; do
export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/rerun-best.py -s $2 -d $3 -g $4 --seed ${seed} --pinit_model $5 -i 1 --target_model $6
done