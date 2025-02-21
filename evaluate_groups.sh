#!/bin/bash
for groupname in news qmsumm ectsum scitldr shakespeare sales_email sales_call samsum billsum podcast; do
export PYTHONPATH="./src"; python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d summedits-${groupname}/sync -t summedits-${groupname}/test -e 3
#echo python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d summedits-${groupname}/sync -t summedits-${groupname}/test -e 3
done
