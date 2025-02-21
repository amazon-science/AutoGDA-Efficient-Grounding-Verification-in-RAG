#!/bin/sh

cd ..
rm -r checkpoints/*
python fine_tuning_crossencoder.py --dataset="summedits-news/train" \
       --test_dataset="summedits-news/test" \
       --save_model_name="vectara-summedits-news" \
       --base_model="vectara/hallucination_evaluation_model"  \
       --batch_size=8 \
       --num_epochs=3 \
       --label_col="label_binary" \
       --learning_rate=0.00001

#       --train_layers="classifier pooler deberta.encoder.LayerNorm deberta.encoder.rel_embeddings deberta.encoder.layer.11"

#python inference.py --dataset="summedits-news/test" \
#       --model="vectara-summedits-news" \
#       --batch_size=2 \
#       --label_col="label_binary"



rm -r checkpoints/*
python inference.py --dataset="summedits-news/test" \
       --model="vectara-summedits-news" \
       --batch_size=2 \
       --label_col="label_binary"
