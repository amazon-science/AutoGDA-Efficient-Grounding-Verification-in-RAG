#!/bin/bash
for groupname in news qmsumm ectsum scitldr sales_email samsum podcast; do #shakespeare billsum
export PYTHONPATH="."; python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d summedits-${groupname}/train -t summedits-${groupname}/train -e 0 --save_model_name summedits-${groupname}-testevalstrat
#echo python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d summedits-${groupname}/sync -t summedits-${groupname}/test -e 3
done
