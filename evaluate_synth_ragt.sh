#!/bin/bash
groupname=all
export PYTHONPATH="."; python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d ragtruth-${groupname}/trainsync -t ragtruth-${groupname}/test --save_model_name ragtruth-${groupname}-finetune-long1280snyc -e 3 -m "tasksource/deberta-base-long-nli"
export PYTHONPATH="."; python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d ragtruth-${groupname}/trainsync -t ragtruth-${groupname}/test --save_model_name ragtruth-${groupname}-finetune-long1280vsnyc -e 3
