#!/bin/bash
for ((i = 3; i <= $#; i++ )); do
  echo "Using missmodel: ${!i} on device $1 in version $2"
  export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/rmiss_eval3.py --rmiss_model ${!i} --seed_version $2
done