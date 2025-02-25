#!/bin/sh

GROUPS="news podcast billsum samsum sales_call sales_email shakespeare scitldr qmsumm ectsum"

cd ..
rm -r checkpoints/*
python fine_tuning_crossencoder.py --dataset="Salesforce/summedits/train" \
       --test_dataset="Salesforce/summedits/test" \
       --save_model_name="vectara-summedits" \
       --base_model="vectara/hallucination_evaluation_model"  \
       --batch_size=8 \
       --num_epochs=3 \
       --label_col="label_binary" \
       --learning_rate=0.00001

for g in $GROUPS
do
  echo "Processing $g" # always double quote "$f" filename
  python inference.py --dataset="summedits-$g/test" \
       --model="vectara-summedits" \
       --batch_size=2 \
       --label_col="label_binary"
done