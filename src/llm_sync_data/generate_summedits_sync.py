"""
"""
import sys
import os
import json
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import argparse
import time
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from src.utils.s3 import upload_csv_to_s3
from src.utils.constants import _BUCKET_NAME
from src.utils.bedrock_utils import get_bedrock_batch_response
from src.utils.script_utils import get_datasets

def generate_prompt_request(system_prompt: str, documents: List[str]) -> List[str]:
    """
    Format the LLM prompts for this batch.
    """

    generate_prompt = lambda prefix, doc: f"""
        {prefix} 

        The document is: 
        
         <document>
         {doc}
         </document>

         Assistant: """

    prompts = [
        generate_prompt(system_prompt, doc) for doc in documents
    ]

    return prompts

def extract_tag(llm_response: str, tag_name: str) -> str:
    pos_start = llm_response.find(f'<{tag_name}>')
    pos_end = llm_response.find(f'</{tag_name}>')
    if pos_start  == -1 or pos_end == -1: return None

    return llm_response[pos_start+len(tag_name)+2:pos_end]

def construct_prompt_few_shot(df_org, evidence_use, n_few_shot=3, n_generate=3, entailment=True, mode="summ"):
    FEWSHOT_PROMPT = "<example {}>{}</example {}>\n"

    if mode == "summ":
        START_PROMPT = """\n\nHuman: You are given the following document wrapped in <document> </document> tags:
               <document>{}</document> \
    Your task is to generate summaries from a document. Here are some examples of how the summaries could look like. \
    Note however that some of the samples contain incorrect information that is not part of the document! \
    Here are the examples:\n"""

        FACTUAL_PROMPT = """Now your task is to generate {} summaries from the document. However, unlike some of the examples \
    given above, **the summaries must be entirely supported by the document.** Only include information that is \
    directly inferrable from the document. It is also important that the summaries reflect the style, length and wording of examples. \
    If there are common patterns or sentence structures in the examples summaries, the created summaries should reflect those. Each summary is identified with an integer from 0 to {}. \
    The summaries must be wrapped in <summary #></summary #> tags, where # is replaced with the summary id. Assistant:\n\n"""

        NONFACTUAL_PROMPT = """
             Your task is to generate {} summaries from the document. However, now all of the summaries must contain at least one piece of non-factual information. This
             can be some information that is not present in the document or some information that is contradictory to the information in the document, but intuitively appears to make sense.
             Otherwise they reflect the style, length and wording of examples. If there are common patterns or sentence structures in the examples summaries, the created summaries should reflect those. 
             Modify different pieces of information at different places in the document.
             Each summary is identified with an integer from 0 to {}. 
             The summaries must be wrapped in <summary #></summary #> tags, where # is replaced with the summary id. Assistant:\n\n"""
    elif mode == "qa":
        START_PROMPT = """\n\nHuman: You are given the following instructions which contain a question and some information in <document> </document> tags:
                       <document>{}</document> \n \
            Your task is to generate useful responses from the document. Here are some examples of how the responses could look like. \
            Note however that some of the samples contain incorrect information that is not part of the document! \
            Here are the examples:\n"""

        FACTUAL_PROMPT = """Now your task is to generate {} responses for the question using the document. However, unlike some of the examples \
            given above, **the responses must be entirely supported by the document.** Only include information that is \
            directly inferrable from the document. It is also important that the responses reflect the style, length and wording of examples. \
            If there are common patterns or sentence structures in the examples responses, the created responses should reflect those. Each response is identified with an integer from 0 to {}. \
            The responses must be wrapped in <response #></response #> tags, where # is replaced with the response id. Assistant:\n\n"""

        NONFACTUAL_PROMPT = """
             Your task is to generate {} responses for the question using the information in the document. However, now all of the replies must contain at least one piece of non-factual information (even if the document in tags may state the opposite). This
             can be some information that is not present in the document or some information that is contradictory to the information in the document, but intuitively appears to make sense. \
             Otherwise they reflect the style, length and wording of examples. If there are common patterns or sentence structures in the example responses, the created responses should reflect those. \
             Modify different pieces of information at different places in the document.
             Each responses is identified with an integer from 0 to {}. 
             The responses must be wrapped in <response #></response #> tags, where # is replaced with the response id. Assistant:\n\n"""

    elif mode == "qa2":
        START_PROMPT = """\n\nHuman: You are given the following instructions which contain a question and some background information on the question in <information> </information> tags:
                       <information>{}</information> \n \
            Your task is to generate useful responses from the background information. Here are some example(s) of how the responses could look like. \
            Note however that some of the examples contain incorrect facts that are not part of the given background information! \
            Here are the example(s):\n"""

        FACTUAL_PROMPT = """Now your task is to generate {} responses for the question using the given  background information. However, unlike some of the examples \
            given above, **the responses must be entirely supported by the given background information.** Only include statements that are \
            directly inferrable from the background information. It is also important that the responses reflect the style, length and wording of examples. \
            If there are common patterns or sentence structures in the example responses, the created responses should reflect those. Each response is identified with an integer from 0 to {}. \
            The responses must be wrapped in <response #></response #> tags, where # is replaced with the response id. Assistant:\n\n"""

        NONFACTUAL_PROMPT = """
             Your task is to generate {} responses for the question using the given background information. However, now all of the replies must contain at least one piece of non-factual information. This
             can be some facts that are not inferrable or present in the background information or some information that is contradictory to the background information given, but intuitively appears to make sense. \
             Otherwise they reflect the style, length and wording of examples. If there are common patterns or sentence structures in the example responses, the created responses should reflect those. \
             Modify different pieces of the background information at different places.
             Each response is identified with an integer from 0 to {}. 
             The responses must be wrapped in <response #></response #> tags, where # is replaced with the response id. Assistant:\n\n"""


    prompt_base = START_PROMPT.format(evidence_use)
    max_samples = len(df_org[df_org["evidence"] == evidence_use])
    examples_random = df_org[df_org["evidence"] == evidence_use]["claim"].sample(min(max_samples, n_few_shot)).values
    for i in range(len(examples_random)):
        prompt_base += FEWSHOT_PROMPT.format(i, examples_random[i], i)
    if entailment:
        prompt = prompt_base + FACTUAL_PROMPT.format(n_generate, n_generate - 1)
    else:
        prompt = prompt_base + NONFACTUAL_PROMPT.format(n_generate, n_generate - 1)
    return prompt

def _execute_promts(prompts, prompt_info, model_id, mode="summ", max_retries=2):
    #print(len(prompts))
    n_expected = sum(p[3] for p in prompt_info)
    sync_dic = {'id': [], 'group': [], 'query': [], 'evidence': [], 'claim': [], 'label': [],
                'label_binary': []}
    n_retries = 0
    while len(sync_dic["id"]) < n_expected:
        if n_retries > max_retries:
            print(f"Max retries exceeded returning {len(sync_dic['id'])} of {n_expected}.")
            break
        n_retries += 1
        sync_dic = {'id': [], 'group': [], 'query': [], 'evidence': [], 'claim': [], 'label': [],
                    'label_binary': []}
        llm_responses = get_bedrock_batch_response(prompts, model=model_id, temperature=0.5)
        # print(llm_responses)
        for batch_nr, resp in enumerate(llm_responses):
            group, evidence_use, label, n_target = prompt_info[batch_nr]
            for summ_id in range(n_target):
                # for i in range(len(doc_batch)):
                claim = extract_tag(resp, f'{"summary" if mode=="summ" else "response"} {summ_id}')
                LABEL = "ENTAILMENT" if label else "NEUTRAL"
                if claim is not None:
                    sync_dic['id'].append(len(sync_dic['id']))
                    sync_dic['group'].append(group)
                    sync_dic['evidence'].append(evidence_use)
                    sync_dic['query'].append('Summarize')
                    sync_dic['claim'].append(claim.strip(" ").strip("\n"))
                    sync_dic['label'].append(LABEL)
                    sync_dic['label_binary'].append(int(LABEL == 'ENTAILMENT'))
                else:
                    print(f"Invalid format. Retrying {batch_nr}, {summ_id}")
    return pd.DataFrame(sync_dic)

def few_shot_prompting(df_org, n_generate=6, n_few_shot=3, batch_size=5, model_name='claude3-haiku',
                       simultanous_prompts=32, min_req_examples=2, prompt_mode="summ"):
    """ Generate a few shot prompt for synthetic data.
        df_org: dataframe with the original data, must contain the evidences and example claims.
        min_req_examples: Minimum examples for a piece of evidence that are required for the generation. Otherwise will be skipped.
        simultanous_prompts: How many prompts to send to the LLM Backend at once.
    """
    if model_name == 'claude3-sonnet':
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    elif model_name == 'claude3-haiku':
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    elif model_name in ["gpt-3.5-turbo", "gpt-4-mini", "gpt-4o-mini"]:
        model_id = model_name
    else:
        raise ValueError(f"Unknown model name: {model_name}.")

    evidence_items = df_org["evidence"].unique()

    df_list = []
    prompts = []
    prompt_data = []
      # keep track of expected t
    print("min req", min_req_examples)
    for evidence_use in tqdm(evidence_items, desc='Getting samples for each evidence.'):
        if len(df_org[df_org["evidence"] == evidence_use]) < min_req_examples:
            print("Not enough claims for evidence. Skipping.")
            continue
        for label in [True, False]:
            group = df_org[df_org["evidence"] == evidence_use]["group"].iloc[0]
            for i in range(n_generate // batch_size):
                prompts.append(construct_prompt_few_shot(df_org, n_few_shot=n_few_shot, evidence_use=evidence_use, n_generate=batch_size, entailment=label, mode=prompt_mode))
                prompt_data.append((group, evidence_use, label, batch_size))
            if n_generate % batch_size > 0:
                prompts.append(construct_prompt_few_shot(df_org, n_few_shot=n_few_shot, evidence_use=evidence_use, n_generate=n_generate % batch_size, entailment=label, mode=prompt_mode))
                prompt_data.append((group, evidence_use, label, n_generate % batch_size))

            if len(prompts) < simultanous_prompts: ## Room for more prompts?
                continue

            df_list.append(_execute_promts(prompts, prompt_data, model_id=model_id, mode=prompt_mode))
            prompts = []
            prompt_data = []

    if len(prompts) > 0:
        #print(f"executing {len(prompts)} final prompts.")
        df_list.append(_execute_promts(prompts, prompt_data, model_id=model_id, mode=prompt_mode))
    sync_df = pd.concat(df_list, ignore_index=True, axis=0)
    return sync_df


def get_queries_hypothesis_and_labels(documents: pd.Series, model_id: str, num_summaries: int, group='None') -> pd.DataFrame:
    BASE_PROMPT = f"""
    Human: You will be given a text document wrapped in <document></document> tags. 
    Your task is to generate  {num_summaries} short summaries (each less than 50 words) from the document. 
    The summaries must be semantically different from each other.  
    The summaries must be diverse in the topics they cover and the length of the summaries. 
    Each summary is identified with an integer from 0 to {num_summaries-1}. 
    The summaries must be wrapped in <summary #></summary #> tags, where # is replaced with the summary id.   
    """

    FACTUAL_PROMPT = f"""
    {BASE_PROMPT}

    The summaries must be supported by the given document. Do not add any information to the summaries 
    that is not found in the document. Moreover, the summaries must not contain any claims that contradict the information in the
    text document. 
    """

    NEUTRAL_PROMPT = f"""
    {BASE_PROMPT}
    
    The summaries must not be entirely supported by the document. This means that the summaries must contain at
    least one claim that is not supported by the document (that is, the summaries must hallucinate). However, 
    the final summaries should still appear correct and supported by the document to the human eye and have less than 50 words. 
    """
    N = len(documents)
    batch_size  = 8
    sync_dic = {'id': [], 'group': [], 'query': [], 'evidence': [], 'claim': [], 'label': [], 'label_binary': []}


    # ------------------ #
    # Get responses
    # ------------------ #
    for LABEL, PROMPT  in [
            ('NEUTRAL', NEUTRAL_PROMPT),
            ('ENTAILMENT', FACTUAL_PROMPT)
        ]:

        for doc_batch in tqdm(np.array_split(documents, N // batch_size), desc=f'Getting {LABEL} samples.'):
            prompts = generate_prompt_request(PROMPT, doc_batch)
            llm_responses = get_bedrock_batch_response(prompts, model=model_id, temperature=0.0)

            for summ_id in range(num_summaries):

                for doc, res in zip(doc_batch, llm_responses):
                # for i in range(len(doc_batch)):
                    claim = extract_tag(res, f'summary {summ_id}')
                    if claim is not None:
                        sync_dic['id'].append(len(sync_dic['id']))
                        sync_dic['group'].append(group)

                        sync_dic['evidence'].append(doc)
                        sync_dic['query'].append('Summarize')
                        sync_dic['claim'].append(claim)
                        sync_dic['label'].append(LABEL)
                        sync_dic['label_binary'].append(int(LABEL == 'ENTAILMENT'))

    sync_df = pd.DataFrame(sync_dic)
    return sync_df

if __name__ == "__main__":
    """
    Generates synthetic data from an existing dataset for premises. 
    For each premise it generates a hypothesis and a label from an LLM. 
    """
    os.environ["AWS_REGION_NAME"] = 'us-east-1'
    parser = argparse.ArgumentParser(prog='LLM api', description='', epilog='')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-m', '--model', choices=['claude3-sonnet', 'claude3-haiku'], default='claude3-haiku')
    # parser.add_argument('-p', '--prompt_name', choices=list(PROMPTS.keys()), default='base_prompt')
    parser.add_argument('-r', '--region_name', choices=['us-east-1', 'us-east-2', 'us-west-1'], default='us-east-1')
    parser.add_argument('-d', '--dataset', type=str, default="ragtruth")
    parser.add_argument('-g', '--group', choices=["Summary", "QA"], default="QA")
    parser.add_argument('-s', '--split', choices=["train", "test", "val"], default="train")
    parser.add_argument('--n_generate', type=int, default=16, help="number of examples to generate per evidence")

    args = parser.parse_args()
    #NUM_SUMMARIES_PER_PREMISE = 50

    model_name = args.model
    os.environ["AWS_REGION_NAME"] = 'us-east-1'
    if model_name == 'claude3-sonnet':
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    if model_name == 'claude3-haiku':
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"


    # --------------- #
    # Load base data
    # --------------- #
    dtrain, dtest, dval  = get_datasets(args.dataset, args.group)
    ddict = {"train": dtrain, "test": dtest, "val": dval}
    df = ddict[args.split].df
    os.makedirs("sync_data", exist_ok=True)
    print('Number of premises: ', len(df))

    #for g in df['group'].unique():
    #    df_temp = df[df['group'] == g]
    #    print(f'Generating group = ', g)
    #    premises = df_temp['evidence'].unique()
    #    print('premises: ', len(premises))
    #    sync_df = get_queries_hypothesis_and_labels(premises, model_id, NUM_SUMMARIES_PER_PREMISE, group=g)
    #    path = f'sync_data/summedits-sync-{g}.csv'
    #    print(f'Saving {path}')
    #    upload_csv_to_s3(sync_df, _BUCKET_NAME, path)
    sync_df = few_shot_prompting(df, n_generate=args.n_generate, n_few_shot=3, batch_size=4, model_name=model_id, mode="summ" if args.group=="Summary" else "qa2")
    path = f'sync_data/d16_{dataset_name.replace("/", "-")}.csv'
    sync_df.to_csv(path, index=False)
    upload_csv_to_s3(sync_df, _BUCKET_NAME, path)