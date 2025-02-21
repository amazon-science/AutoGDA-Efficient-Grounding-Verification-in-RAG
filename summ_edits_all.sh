#!/bin/bash
for groupname in news qmsumm ectsum scitldr shakespeare sales_email sales_call samsum billsum podcast; do
export PYTHONPATH="."; export TOKENIZERS_PARALLELISM=false; python3 src/sync_data/summedits_loop.py ${groupname}
#echo python3 src/cross_encoder_model/fine_tuning_crossencoder.py -d summedits-${groupname}/sync -t summedits-${groupname}/test -e 3
done
