#!/bin/bash
# call args: device_no study dataset group init_model target_model
for seed in 1 2 3 4 5; do
export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/rerun-best.py --seed ${seed} --study $2 -d $3 -g $4 \
--pinit_model $5 --no_reeval -i 2 --run_folder_suffix "_randomsel" --config_file "config_files/hyperparameter_random_selector.json"
done