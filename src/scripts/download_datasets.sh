#!/bin/bash

## Download LFQA-Veri
mkdir -p data/lfqa-verification/annotations
wget https://raw.githubusercontent.com/timchen0618/LFQA-Verification/refs/heads/main/data/annotations/annotations-alpaca_wdoc.json -P data/lfqa-verification/annotations
wget https://raw.githubusercontent.com/timchen0618/LFQA-Verification/refs/heads/main/data/annotations/annotations-gpt3_wdoc.json -P data/lfqa-verification/annotations
wget https://raw.githubusercontent.com/timchen0618/LFQA-Verification/refs/heads/main/data/annotations/annotations-gpt3_whudoc.json -P data/lfqa-verification/annotations
wget https://raw.githubusercontent.com/timchen0618/LFQA-Verification/refs/heads/main/data/annotations/annotations-webgpt.json -P data/lfqa-verification/annotations
mkdir -p data/lfqa-verification/docs
wget https://raw.githubusercontent.com/timchen0618/LFQA-Verification/refs/heads/main/data/docs/docs-webgpt.json -P data/lfqa-verification/docs
wget https://raw.githubusercontent.com/timchen0618/LFQA-Verification/refs/heads/main/data/docs/docs-human.json -P data/lfqa-verification/docs

## Download RAGTruth
mkdir -p data/ragtruth-data/dataset
wget https://raw.githubusercontent.com/ParticleMedia/RAGTruth/refs/heads//main/dataset/response.jsonl -P data/ragtruth-data/dataset
wget https://raw.githubusercontent.com/ParticleMedia/RAGTruth//refs/heads/main/dataset/source_info.jsonl -P data/ragtruth-data/dataset
