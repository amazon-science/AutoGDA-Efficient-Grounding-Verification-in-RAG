#!/bin/bash
for groupname in mistral-7b gpt-3.5 llama-7b llama-13b llama-70b gpt-4 all; do #
export PYTHONPATH="."; python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d ragtruth-${groupname}/train -t ragtruth-${groupname}/test --save_model_name ragtruth-${groupname}-finetune-long1280v -e 3
export PYTHONPATH="."; python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d ragtruth-${groupname}/train -t ragtruth-${groupname}/test --save_model_name ragtruth-${groupname}-finetune-long1280 -e 3 -m "tasksource/deberta-base-long-nli"
#export PYTHONPATH="."; python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d ragtruth-${groupname}/train -t ragtruth-${groupname}/test --save_model_name ragtruth-${groupname}-finetune-long512b -e 3 -m "facebook/bart-large-mnli"
#echo python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d summedits-${groupname}/sync -t summedits-${groupname}/test -e 3
done
