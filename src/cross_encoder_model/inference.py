import sys
import os
import time
import pandas as pd
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader,Dataset
from typing import List
import nltk
import json
nltk.download('punkt')

sys.path.append('../')
from utils import evaluation_utils as eval
from utils.data_utils import AnnotatedTextDataset, DATA_LOADER_FN
from utils.s3 import upload_csv_to_s3
from utils.model_s3_utils import load_cross_encoder_model_from_s3
from utils.hallucination_scores_utils import get_hallucination_scores
from utils.global_variables import _BUCKET_NAME, MODELS, COLOR


def get_num_tokens_batch(model, evidence: List[str], claim: List[str]):
    # texts = [[evidence, claim]]
    texts = list(zip(evidence, claim))
    tokenized = model.tokenizer(
        texts,  return_tensors="np"
    )
    # token_sizes = [len(t['input_ids']) for t in tokenized]
    token_sizes = [len(t) for t in tokenized['input_ids']]
    # num_tokens = len(tokenized['input_ids'][0])
    return token_sizes


def get_num_tokens(model, evidence: str, claim: str):
    texts = [[evidence, claim]]
    tokenized = model.tokenizer(
        texts,  return_tensors="np"
    )
    num_tokens = len(tokenized['input_ids'][0])
    return num_tokens

def split_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    sentences = [sent for sent in sentences if len(sent) > 10]
    return sentences

def ranker(model, evidence_docs: List, claims: List):
    new_evidence_docs = []
    for evidence, claim in zip(evidence_docs, claims):
        # 1) Check in the evidence claim fit into 512 tokens
        num_tokens = get_num_tokens(model, evidence, claim)
        if num_tokens > 512:
            # 2) Rank the inputs.
            evidence_sentences = split_sentences(evidence)
            num_sentences = len(evidence_sentences)
            in_sentence = np.zeros(num_sentences)
            # 1. We rank all sentences in the evidence for the claim
            ranks = model.rank(claim, evidence_sentences)
            # Build The new evidence document with the most relevant sentences
            sentences_concat = ""
            for rank in ranks:
                sentence_id = rank['corpus_id']
                rel_evidence_sentence =  evidence_sentences[sentence_id]
                sentences_concat += f" {rel_evidence_sentence}"
                num_tokens = get_num_tokens(model, sentences_concat, claim)

                if num_tokens > 512:
                    # New sentence is over limit
                    break
                in_sentence[sentence_id] = 1  # Include sentence into evidence

            # Add the most relevant setences in order
            relevant_sentences = []
            for i in range(num_sentences):
                if in_sentence[i]:
                    relevant_sentences.append(evidence_sentences[i])

            new_doc = " ".join(relevant_sentences)
            # Add retracted document to the evidence list
            new_evidence_docs.append(new_doc)

        else:
            # Evidence doc is under the threshold
            new_evidence_docs.append(evidence)

    return new_evidence_docs





if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Transformers models', description='', epilog='')
    parser.add_argument('-d', '--dataset', default='Salesforce/summedits/test', choices=list(DATA_LOADER_FN.keys()))
    parser.add_argument('-m', '--model', default='vectara/hallucination_evaluation_model')
    parser.add_argument('--label_col', default='label_binary')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument("--rank", action='store_true')
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    print(args)

    dataset_name = args.dataset
    batch_size = args.batch_size
    model_name = args.model
    label_col = args.label_col
    rank_context = args.rank
    device = args.device

    # model_path = f"{base_model_name}-{data_id}"
    print(f'Running {model_name} on dataset {dataset_name} ')
    print(f'batch_size = {batch_size}')

    #######
    ## Load data
    #######
    dataset = DATA_LOADER_FN[dataset_name]()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    ##################
    ## Load base model
    ##################
    # if model_name.startswith('che')

    num_labels = 1  if label_col == "label_binary" else 3
    try:
        model = CrossEncoder(model_name, num_labels=num_labels, device=device)
        print(f'Loading hugginface model.')
    except:
        # Get from s3
        model_path = f"checkpoints/{model_name}"
        print(f'Loading {model_path}')
        load_cross_encoder_model_from_s3(_BUCKET_NAME, model_path, model_path)
        model = CrossEncoder(model_path, num_labels=num_labels, device=device)
        print(f'Loading s3 model.')



    def get_scores(evidences_batch: List[str], claims_batch: List[str]) -> pd.DataFrame:
        tokens = get_num_tokens_batch(model, evidences_batch, claims_batch)
        if rank_context:
            # Rank evidence sentences by relevance to the claim
            evidences_batch = ranker(model, evidences_batch, claims_batch)
        scores = model.predict(list(zip(evidences_batch, claims_batch)))
        samples = []
        return pd.DataFrame({'scores': scores, 'tokens': tokens})

    ##############
    ##  Run model
    ##############
    data_id = dataset.get_dataset_identifier()
    stime = time.time()
    results_df = get_hallucination_scores(get_scores, dataloader)
    elapsed_time = time.time() - stime
    print(f'Elapsed time: {elapsed_time:.4f}')

    all_scores = np.array(results_df['scores'].values)
    all_labels = 1* np.array(dataset.df[label_col].values)
    res = eval.all_metrics(all_labels, all_scores, buckets=20)
    eval.print_metric_results(res)

    save_path = f'results/{model_name}/{data_id}/results.csv'
    if rank_context:
        model_name = f"{model_name}_rank"
    save_path = f'results/{model_name}/{data_id}/results.csv'
    os.makedirs(f'results/{model_name}/{data_id}', exist_ok=True)

    print('Saving local copy in', save_path)
    results_df.to_csv(save_path)

    print(f'Saving remote copy in {_BUCKET_NAME}/{save_path}')
    upload_csv_to_s3(results_df, _BUCKET_NAME, save_path)