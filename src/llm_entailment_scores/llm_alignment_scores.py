"""
"""
import sys
import os
import json
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import argparse
import time
from typing import List
import numpy as np
from copy import deepcopy

sys.path.append('../')
from src.utils.bedrock_utils import get_bedrock_batch_response
#from src.utils import evaluation_utils as eval
import time
from tqdm import tqdm
from openai import OpenAI
import os
import json
from multiprocessing import Pool



"""
The prompt ask the LLM for a binary output with the confidence score.
"""

SELFCHECK_PROMPT = f"""\n\nHuman:
You are an assistant that will be given a claim and a corresponding evidence document and your job is to detect if 
the claim is supported by the evidence. In other words, your task is to help the user understand if the given claim
is factually consistent with the evidence provided. You will output YES if the claim is completely supported by the
evidence document and NO otherwise. An output of NO, means the the claim contains information that is not supported 
by the evidence (hallucinated) or contains information that is contradicted by the evidence.
 Your answer must be wrapped in <answer></answer> tags. 
The evidence will come first and be wrapped in <evidence></evidence> tags and the claim will be wrapped in <claim></claim> tags. 

Your output must include three components (in the same order): 

1) Reasoning step: The first output you will produce is a reasoning step, which will be wrapped  in <reasoning></reasoning> tags.
For this step, you must break down the given claim into verifiable subclaims. For example, if the claim is
"NYC has a population over 8 million people, making the largest city in the U.S.", then the  subclaims are
"NYC is a city in the U.S.", "NYC has a population of 8 million people"  and "NYC is the largets city in the U.S.".
For each subclaim provide a label of YES or NO. Moreover, provide your reasoning 
behind your choice and cite the supporting evidence. Separate each subclaim with two line breaks. 

2) Answer: Then you must provide your final answer for the entire claim wrapped in <answer></answer> tags. 
 Only output YES or NO. The answer should be YES if the entire claim is supported by the evidence.

3) Confidence score:  Moreover, you will provide a real-valued score between 0 and 1, wrapped in 
<score></score> tags, that represents your  confidence that the answer is YES, or that the claim is supported by the 
evidence. A score greater than 0.5 means you believe the answer is YES and a score less than 0.5 means you 
believe the answer is NO. A score of 0.5 means you are uncertain of the answer. 
A score of 1.0, means high confidence that the claim is supported by the given the evidence.  A score closer
 to 0.0 means there is higher probability that the claim is not supported by the evidence or contradicted by the evidence.  

To guide your answers, you will be provided a few examples.  Each example is wrapped by <example></example>
tags, and  contains five components: The example evidence document (in <evidence></evidence> tags), the
example claim document (in <claim></claim> tags), the ground true answer (wrapped in <answer></answer> tags),  
the confidence score that the claim is supported by the evidence  (in <score></score> tags), and the reasoning behind the answer 
 (in <reasoning></reasoning> tags). 

Here is the format for each example:
<example>
    <evidence></evidence>
    <claim></claim>
    <reasoning></reasoning>
    <answer></answer>
    <score></score>
</example>

Here are the examples: 

Example 1:
    <example>
        <evidence>The CEO of Amazon is Andy Jassy</evidence>
        <claim>The CEO of Amazon is Adam Selipsky</claim>
        <reasoning>
       Adam Selipsky is a CEO -- NO -- based on the evidence we don't know if Adam Selipsky is a CEO.


       The CEO of Amazon is Adam Selipsky -- NO -- the evidence says the the CEO of Amazon is  Andy Jassy, contradicting the claim that the CEO of Amazon is Adam Selipsky. 
        </reasoning>
        <score>0.0</score>
        <answer>NO</answer>
    </example>

Example 2:
    <example>
        <evidence>The CEO of Amazon is Andy Jassy</evidence>
        <claim>Andy Jassy is not just a CEO, but a very good tennis player</claim>
        <reasoning>
       Andy Jassy is a CEO -- YES -- The evidence says that Andy Jassy is a CEO.


       Andy Jassy is a very good tennis player  -- NO -- The evidence never says that Andy Jassy is a tennis player.
        </reasoning>
        <score>0.0</score>
        <answer>NO</answer>
    </example>

Example 3:
    <example>
        <evidence>NYC is the largest city in the U.S.</evidence>
        <claim>NYC is in the U.S.</claim>
        <reasoning> 
        NYC is in the U.S. -- YES -- this subclaim is backed by the evidence.


        </reasoning>
        <score>1.0</score>
        <answer>YES</answer>
    </example>

Example 4:
    <example>
        <evidence>NYC is the city with the largest population in the U.S.</evidence>
        <claim>NYC has a population of 8 million people, more than any other city in the U.S.</claim>
        <reasoning> 
        NYC has a population of 8 million people  -- NO -- The evidence says that NYC has the largest population but it does not mention that it has 8 million people 

        NYC has a larger popultion than any other city -- YES --  this subclaim comes directly from  the evidence  "NYC is the city with the largest population in the U.S."
        </reasoning>
        <score>0.0</score>
        <answer>NO</answer>
    </example>

Example 5:
    <example>
        <evidence>The U.S stock market was up today after the fed announces reductions in the interest rate.  </evidence>
        <claim>The federal reserve will reduce interest rates. </claim>
        <reasoning> 
        The federal reserve will reduce interest rates  -- YES --  The evidence states that the "fed" (ferering to the federal reserve) announced interest rate reductions in the future.  
        </reasoning>
        <score>1.0</score>
        <answer>YES</answer>
    </example> 

Example 6:
    <example>
        <evidence>Nikola Jokić was voted the league's MVP (Most valuable player) of the 2023 NBA season,  after leading the league in scoring and assists.   </evidence>
        <claim>The MVP of 2023 is Nicola Jokić. </claim>
        <reasoning> 
        The MVP of 2023 is Nicola Jokić. -- YES --   The claim states that "Nicola Jokic" is the MVP (probably refering to the most valuable player in the NBA league). But "Nicola Jokic" is probably a typo in the name, and the claim is in fact refering to Nicola Jokić winning the NBA MVP award for the 2023 season, which is supported by the evidence.
        </reasoning>
        <score>0.75</score>
        <answer>YES</answer>
    </example> 
"""

BINARY_PROMPT = f"""\n\nHuman:
You are an assistant that will be given a claim and a corresponding evidence document and your job is to detect if 
the claim is fully supported by the evidence. In other words, your task is to help the user understand if the given claim
is factually consistent with the evidence provided. You will answer YES if the claim is completely supported by the
evidence document and answer NO otherwise. An answer of NO, means the the claim either contains information that is not supported or contained in the
evidence (hallucinated) or contains information that is contradicted by the evidence.
Your final answer must be wrapped in <final></final> tags so it can be either <final>YES</final> or <final>NO</final>. Always provide a final answer!
The evidence will come first and be wrapped in <evidence></evidence> tags and the claim will be wrapped in <claim></claim> tags. 
"""

ZERO_ONE_PROMPT = """Determine whether the provided claim is consistent with the corresponding document. \
Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent.
Document: {}
Claim: {}
Please assess the claim’s consistency with the document by responding with either "1" (consistent) or "0" (inconsistent). Do not output anything else.
Answer: """

MINICHECK_PROMPT = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that \
all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent.
Document: {}
Claim: {}
Please assess the claim’s consistency with the document by responding with either "yes" or "no". Do only output either "yes" or "no"!
Answer: """

def extract_tag(llm_response: str, tag_name: str) -> str:
    pos_start = llm_response.find(f'<{tag_name}>')
    pos_end = llm_response.find(f'</{tag_name}>')
    if pos_start == -1 or pos_end == -1:
        return None
    return llm_response[pos_start + len(tag_name) + 2:pos_end]


def generate_prompt_request(evidence_docs: List[str], claims: List[str], system_prompt: str, zero_one_prompt=False) -> List[str]:
    """
    Format the LLM prompts for this batch.
    """
    if not zero_one_prompt:
        generate_prompt = lambda prefix, evidence, claim: f"""
            {prefix} 
    
            The evidence and claim you will score is: 
             <evidence>{evidence}</evidence>
             <claim>{claim}</claim>
    
             Assistant: """

        prompts = [
            generate_prompt(system_prompt, evidence, claim) for evidence, claim in list(zip(evidence_docs, claims))
        ]
    else:
        prompts = [
            system_prompt.format(evidence, claim) for evidence, claim in list(zip(evidence_docs, claims))
        ]
    return prompts


def get_scores(evidence_docs: List[str], claims: List[str],
               model_id: str,
               system_prompt: str,
               tag_extract="final",
               num_tries=3,
               zero_one_prompt=False,
               temperature = 0.5
               ) -> dict:
    """
    Given a sequence of evidence and claim docs, format the LLM prompt and then calls the API.
    """
    prompts = generate_prompt_request(evidence_docs, claims, system_prompt=system_prompt, zero_one_prompt=zero_one_prompt)
    #print(prompts[0])
    batch_size = len(prompts)
    answers = []
    scores = []
    reasoning = []

    scores = np.zeros((batch_size, num_tries))

    for i in range(num_tries):
        invalid = True
        while invalid:
            invalid = False
            llm_responses = get_bedrock_batch_response(prompts, model=model_id, temperature=temperature)
            for j, llm_response in enumerate(llm_responses):
                if zero_one_prompt:
                    score_str = llm_response.upper()
                else:
                    score_str = extract_tag(llm_response, tag_extract)
                #print("Received:", score_str)
                if score_str.startswith("YES") or score_str.startswith("1"):
                    score_numeric = 1.0
                elif score_str.startswith("NO") or score_str.startswith("0"):
                    score_numeric = 0.0
                else:
                    # Convert to float.
                    #print(f"received_invalid_score {score_str}.")
                    try:
                        score_numeric = float(score_str)
                    except ValueError:
                        invalid = True
                        print("Received invalid LLM Entailment response: ", score_str)
                        continue
                scores[j, i] = score_numeric

    scores_avg = scores.mean(axis=1)
    return {'scores': scores_avg}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='LLM api', description='', epilog='')
    parser.add_argument('-d', '--dataset', choices=list(DATA_LOADER_FN.keys()))
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-m', '--model', choices=['claude3-sonnet', 'claude3-haiku', "llama3"],
                        default='claude3-haiku')
    parser.add_argument('-s', '--num_tries', type=int, default=1)
    parser.add_argument('-r', '--region_name', choices=['us-east-1', 'us-east-2', 'us-west-1'], default='us-east-1')
    args = parser.parse_args()

    print(args)
    dataset_name = args.dataset
    batch_size = args.batch_size
    model_name = args.model
    region_name = args.region_name

    os.environ["AWS_REGION_NAME"] = 'us-east-1'
    if model_name == 'claude3-sonnet':
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    if model_name == 'claude3-haiku':
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    elif model_name == 'llama3':
        model_id = "meta.llama3-70b-instruct-v1:0"

    ##############
    ## Load data
    ##############
    dataset = None
    # dataset.df = dataset.df.sample(n=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f'\n\nRunning {model_name} on dataset {dataset.get_dataset_identifier()} ')
    print(f'\t\tData size = {len(dataset.df)}')
    print(f'\t\tnum_tries = {args.num_tries}')


    ##############
    ## Load model
    ##############
    def hallucination_fn(evidences: List[str], claims: List[str]) -> pd.DataFrame:
        output_dict = get_scores(evidences, claims, model_id=model_id, system_prompt=BINARY_PROMPT,
                                 num_tries=args.num_tries)
        return pd.DataFrame(output_dict)


    ##############
    ##  Run
    ##############
    stime = time.time()
    data_id = dataset.get_dataset_identifier()
    stime = time.time()
    results = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        results.append(hallucination_fn(batch["evidence"], batch["claim"]))
    results_df = pd.concat(results, axis=0, ignore_index=True)
    print(f'Elapsed time = {time.time() - stime}')

    all_scores = np.array(results_df['scores'].values)
    all_labels = 1 * np.array(dataset.df['label_binary'].values)
    dataset.df["llm_scores"] = all_scores
    dataset.df.to_csv(f"llm_labeled_{args.dataset.replace('/','-')}.csv", index=False)

    print(f'**** Metrics ****')
    print("accuracy:",  np.sum(all_scores==all_labels) / len(all_scores))
    #res = eval.all_metrics(all_labels, all_scores, buckets=20)
    #eval.print_metric_results(res)

    # Save runtime.
    #results_df['runtime'] = time.time() - stime

    #save_path = f'results/{model_name}/tries_{args.num_tries}/{data_id}/results.csv'
    #os.makedirs(f'results/{model_name}/tries_{args.num_tries}/{data_id}', exist_ok=True)
    #print('Saving local copy in', save_path)
    #results_df.to_csv(save_path)
    #print(f'Saving remote copy in {_BUCKET_NAME}/{save_path}')
    #upload_csv_to_s3(results_df, _BUCKET_NAME, save_path)

API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
API_LEGACY_ENDPOINT = "https://api.openai.com/v1/completions"

def compute_probas_from_logprobs(choice_logprobs):
    """ Compute the probabilities of chosing the labels 0, 1. """
    p1 = -float("inf")
    p0 = -float("inf")
    for token in choice_logprobs:
        if token.token == "1":
            p1 = token.logprob
        if token.token == "0":
            p0 = token.logprob
    #print("0", p0, "1", p1)
    odds_diff = p1-p0
    if np.isnan(odds_diff):
        p1 = 0.5
    elif odds_diff == float("inf"):
        p1 = 1.0
    elif odds_diff == -float("inf"):
        p1 = 0.0
    else:
        p1 = np.exp(odds_diff)/(np.exp(odds_diff)+1)
    return np.array([1-p1, p1])

class OpenAImodel:
    """ Use the OpenAI API to perform classifications. """

    def __init__(self, target_model="gpt3.5-turbo", prompt_template=ZERO_ONE_PROMPT,
                 openai_keys_file="data/openai-tobias-project.json", n_threads=6):
        """
            target_model: the API to use
            prompt template: the prompt template to use. will be used with (template % sequence) to generate prompt. Should contain the command to just output either "Yes" or "No".
            openai_keys: tuple of (openai_org, openai_apikey)
        """
        self.n_threads = n_threads
        self.model = target_model
        self.prompt_template = prompt_template

        if openai_keys_file is not None:
            credentials = json.load(open(openai_keys_file))
            self.openai_org, self.openai_key = credentials["org"], credentials["key"]
            self.client = OpenAI(
                organization=self.openai_org,
                api_key=self.openai_key
            )
        else:
            self.openai_org, self.openai_key = None, None

    def train(self, ds):
        pass  # No training requred

    def predict_proba(self, input_sequences):
        # print(len(input_sequences))
        # with p as Pool(self.n_threads):
        # f_map = lambda seq: self._predict_proba_single(input_sequences, seq)
        tasklist = list(
            [(self.openai_org, self.openai_key, self.model, self.prompt_template, ev, clm) for ev, clm in input_sequences])
        with Pool(self.n_threads) as p:
            res_array = p.map(predict_proba_single, tasklist)
        return np.stack(res_array, axis=0)[:,1]

    def predict(self, sentence_pairs):
        if self.client is None:
            raise ValueError(
                "Client is not initialized. Please pass an API key to the constructor or load a model file.")
        message_dict = [{
            "role": "system",
            "content": "You are a helpful assistant for fact-checking."
        },
        {
            "role": "user",
            "content": None
        }]
        req_list = []
        for evidence, claim in tqdm(sentence_pairs, total=len(sentence_pairs)):
            req_list.append(predict_proba_single((self.openai_org, self.openai_key, self.model, self.prompt_template, evidence, claim)))
        return np.array(req_list)[:, 1]


def predict_proba_single(arglist):
    openai_org, openai_key, target_model, prompt_template, predict_ev, predict_claim = arglist
    client = OpenAI(
                organization=openai_org,
                api_key =openai_key
            )
    if client is None:
        raise ValueError("Client is not initialized. Please pass an API key to the constructor or load a model file.")

    message_dict = [{
        "role": "system",
        "content": "You are a helpful assistant for fact-checking."
    },
    {
        "role": "user",
        "content": None
    }]

    message_dict[1]["content"] = prompt_template.format(predict_ev, predict_claim) #self.prompt_template.format(seq)
    #print(message_dict)
    results = client.chat.completions.create(
        model=target_model,
        messages=message_dict,
        logprobs=True,
        top_logprobs = 5,
        temperature=0.00001,
        seed=1
    )
    return compute_probas_from_logprobs(results.choices[0].logprobs.content[0].top_logprobs)
    #print(results.choices[0])