#!/bin/bash
# export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/ablation_eval.py --ft_model $2 --rmiss_model $2 --seed_version 2
# export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/ablation_eval.py --ft_model $2 --rmiss_model ensemble-mean --seed_version 2 --skip_init_eval true
export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/reevaluate_weighted.py $2 ablation_s2_${2}_${2}_ragtruth_QA ablation_s2_ensemble-mean_${2}_ragtruth_QA
# export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/ablation_eval.py --ft_model $2 --rmiss_model $2 --seed_version 3
# export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/ablation_eval.py --ft_model $2 --rmiss_model ensemble-mean --seed_version 3 --skip_init_eval true