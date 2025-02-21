#!/bin/bash
for groupname in 'Healthcare'; do #'Chemistry', 'VisualArts' 'Engineering', 'Business' 'Architecture' 'Law'  'Psychology' 'Education'
export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=2,3; python3 src/scripts/compute_baselines.py -c src/scripts/baseline_config.json -d expertqa -g ${groupname}
done
