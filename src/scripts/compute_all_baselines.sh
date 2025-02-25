#!/bin/bash
# Compute the baselines for all datasets
export PYTHONPATH="."; python3 src/scripts/compute_baselines.py -c src/scripts/baseline_config.json -d summedits -g all
export PYTHONPATH="."; python3 src/scripts/compute_baselines.py -c src/scripts/baseline_config.json -d lfqa-veri -g all
export PYTHONPATH="."; python3 src/scripts/compute_baselines.py -c src/scripts/baseline_config.json -d ragtruth -g QA
export PYTHONPATH="."; python3 src/scripts/compute_baselines.py -c src/scripts/baseline_config.json -d ragtruth -g Summary