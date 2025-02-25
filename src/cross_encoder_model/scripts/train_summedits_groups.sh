#!/bin/sh

GROUPS="news podcast billsum samsum sales_call sales_email shakespeare scitldr qmsumm ectsum"

cd ..

#for g in $GROUPS
#do
#  rm -r checkpoints/*
#  echo "Processing $g" # always double quote "$f" filename
#
#  python fine_tuning_crossencoder.py --dataset="summedits-$g/train" \
#       --test_dataset="summedits-$g/test" \
#       --save_model_name="vectara-summedits-$g" \
#       --base_model="vectara/hallucination_evaluation_model"  \
#       --batch_size=8 \
#       --num_epochs=10 \
#       --label_col="label_binary" \
#       --learning_rate=0.00001
#done

for g in $GROUPS
do
  rm -r checkpoints/*
  python inference.py --dataset="summedits-$g/test" \
       --model="vectara-summedits-$g" \
       --batch_size=2 \
       --label_col="label_binary"
done
