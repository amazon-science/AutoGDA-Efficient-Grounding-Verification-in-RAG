#!/bin/sh

GROUPS="news podcast billsum samsum sales_call sales_email shakespeare scitldr qmsumm ectsum"


cd ..
for g in $GROUPS
do
  echo "Processing $g" # always double quote "$f" filename
  python inference.py --dataset="summedits-$g/test" \
       --model="vectara-summedits-last.layer" \
       --batch_size=2 \
       --label_col="label_binary"
done

#rm -r checkpoints/*
#python fine_tuning_crossencoder.py --dataset="summedits-news/train" \
#       --test_dataset="summedits-news/test" \
#       --save_model_name="vectara-summedits-news" \
#       --base_model="vectara/hallucination_evaluation_model"  \
#       --batch_size=2 \
#       --num_epochs=5 \
#       --label_col="label_binary" \
#       --learning_rate=0.00007

#       --train_layers="classifier pooler deberta.encoder.LayerNorm deberta.encoder.rel_embeddings deberta.encoder.layer.11"

#python inference.py --dataset="summedits-news/test" \
#       --model="vectara-summedits-news" \
#       --batch_size=2 \
#       --label_col="label_binary"




