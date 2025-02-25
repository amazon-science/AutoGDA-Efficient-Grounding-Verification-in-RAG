#!/bin/bash
for ((i = 3; i <= $#; i++ )); do # syntax: device seed rmiss1 rmiss2, rmiss3...
  echo "Using missmodel: ${!i} on device $1 in version $2"
  export PYTHONPATH="."; export CUDA_VISIBLE_DEVICES=$1; python3 src/scripts/rmiss_eval3.py --rmiss_model ${!i} --seed $2
done