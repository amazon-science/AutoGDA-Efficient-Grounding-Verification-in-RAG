## Use different models to compute entailment between passages.
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder
from src.cross_encoder_model.model_wrappers import TwoWayDebertaV2, TwoWayBart, OneWayCrossEncoder, DataCollatorWithTokenization, TwoWayT5
import os
from src.llm_entailment_scores.llm_alignment_scores import get_scores, BINARY_PROMPT, SELFCHECK_PROMPT, MINICHECK_PROMPT
from src.cross_encoder_model.bedrock_guardrail import Guardrail
from typing import List, Tuple, Dict
import torch
from tqdm import tqdm
from src.utils.minicheck import MiniCheck
import spacy
from collections import Counter
from src.llm_entailment_scores.llm_alignment_scores import OpenAImodel
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from vectara.modeling_vectara import HHEMv2ForSequenceClassification
nlp = spacy.load('en_core_web_sm')

@contextmanager
def suppress():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            with redirect_stderr(null):
                yield

def split_sentences(text):
    tokens = nlp(text)
    ret_list = []
    part = ""
    open_quotes = False
    for sent in tokens.sents:
        str_sent = str(sent)
        part += str_sent
        if len(str_sent) < 5:  # shorest sentence: I am.
            continue

        ## count number of quotes
        cnt = Counter(str_sent)
        if cnt['"'] == 1:
            if not open_quotes:
                open_quotes = True
                continue
            else:
                open_quotes = False

        if not open_quotes:
            ret_list.append(part.strip("\n"))
            part = ""
        else:
            if len(part) > 150:  # force split
                ret_list.append(part.strip("\n"))
                part = ""
                open_quotes = False
    if part != "":
        ret_list.append(part.strip("\n"))
    return ret_list

def wrap_tqdm(iterable, use_tqdm=True):
    if use_tqdm:
        return tqdm(iterable, desc="Compute Entailments.")
    else:
        return iterable


class EntailmentCheckModel:
    def __init__(self, model_name_str, batch_size=4, device="cuda", **kvargs):
        """ Load the model.
            Currently supported models are: vectarav1, vectarav2, tasksource,
            bart-large, claude-haiku, claude-sonnet, alignscore, algnscore-large
        """
        self.batch_size = batch_size
        self.model, self.tokenizer = None, None
        self.noise = 0.0  # Additional noise can be added for ablation on p_miss quality.
        ## noise?
        if "noise" in kvargs:
            self.noise = kvargs["noise"]
        if "," in model_name_str:
            parts = model_name_str.split(",")
            self.model_name_str = parts[0]
            for p in parts[1:]: # additional args
                argname, argval = p.split("=")
                if argname == "noise":
                    self.noise = float(argval)
        else:
            self.model_name_str = model_name_str
        self.device = device
        self.kvargs = kvargs

        if self.model_name_str in ['vectara_v1', 'vectara_v2', 'nli-deberta-v3-base', 'tasksource', 'tasksource_v1', 'bart-large', 'flan-t5-base']:
            self.model, self.tokenizer = self._get_model_tok_for_finetunable_models(self.model_name_str)
        elif self.model_name_str == "ensemble-mean" or self.model_name_str == "ensemble-median":
            ## Setup an ensemble of models.
            self.model = [EntailmentCheckModel("bart-large", batch_size= self.batch_size//2),
                          EntailmentCheckModel("tasksource_v1", batch_size= self.batch_size//2),
                          EntailmentCheckModel("alignscore",  batch_size= self.batch_size//2)]
        elif self.model_name_str == 'claude3-haiku' or model_name_str == 'claude3-sonnet' or model_name_str == 'llama3':
            os.environ["AWS_REGION_NAME"] = 'us-east-1'
            if self.model_name_str == 'claude3-sonnet':
                self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            if self.model_name_str == 'claude3-haiku':
                self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
            elif self.model_name_str == 'llama3':
                self.model_id = "meta.llama3-70b-instruct-v1:0"
        elif self.model_name_str in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
            self.model = OpenAImodel(self.model_name_str)
        elif self.model_name_str == "guardrails":
            self.model = Guardrail()
        elif self.model_name_str == "minicheck-t5":
            self.model = MiniCheck(model_name='flan-t5-large')
        elif self.model_name_str == "alignscore-large" or self.model_name_str == "alignscore":
            from alignscore import AlignScore
            if self.model_name_str == "alignscore":
                self.model = AlignScore(model='roberta-base', batch_size=self.batch_size, device=self.device,
                                        ckpt_path="../AlignScore/AlignScore/checkpoints/AlignScore-base.ckpt",
                                        evaluation_mode='nli_sp')
            else: #Roberta large
                self.model = AlignScore(model='roberta-large', batch_size=self.batch_size, device=self.device,
                                    ckpt_path="../AlignScore/AlignScore/checkpoints/AlignScore-large.ckpt",
                                    evaluation_mode='nli_sp')
        else:
            raise ValueError(f"Unknown model name {self.model_name_str}")

    def _get_model_tok_for_finetunable_models(self, model_string):
        if model_string == 'vectara_v1':
            model = CrossEncoder(
                'vectara/hallucination_evaluation_model',
                num_labels=1,
                automodel_args={"ignore_mismatched_sizes": True},
                device=self.device,
                revision="hhem-1.0-open"
            )
            tokenizer = None
        elif model_string == 'nli-deberta-v3-base':
            model = OneWayCrossEncoder(
                'cross-encoder/nli-deberta-v3-base',
                num_labels=3,
                device=self.device,
                automodel_args={"ignore_mismatched_sizes": True},
            )
            tokenizer = None
        elif model_string == 'vectara_v2':
            self.model = HHEMv2ForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model')
            model = self.model.to(self.device)
            tokenizer = self.model.tokenzier
        elif model_string in ['tasksource', 'tasksource_v1']:
            base_model_name = "tasksource/deberta-base-long-nli"
            if self.model_name_str == "tasksource_v1":
                revision = "9f72a81a6be78ec99f78314c4636b3a60259ab30"
            else:
                revision = None
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, revision=revision)
            model = TwoWayDebertaV2.from_pretrained(base_model_name, revision=revision).to(self.device)
        elif model_string == "bart-large":
            base_model_name = "facebook/bart-large-mnli"
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = TwoWayBart.from_pretrained(base_model_name).to(self.device)
        elif model_string == "flan-t5-base":
            base_model_name = "sjrhuschlee/flan-t5-base-mnli"
            model = TwoWayT5.from_pretrained(base_model_name)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            raise ValueError(f"Unknown finetunable model name {model_string}.")
        return model, tokenizer

    def compute_scores(self, sentence_pairs: List[Tuple[str, str]], show_progress=True) -> np.array:
        raw_scores = self._compute_raw_scores(sentence_pairs, show_progress)
        if self.noise > 0.0:
            raw_scores = raw_scores*(1-self.noise) + self.noise*np.random.rand(len(raw_scores))
        return raw_scores

    def reset_inplace(self):
        """ Reset the current NLI model inplace. """
        new_model, _ = self._get_model_tok_for_finetunable_models(self.model_name_str)
        if self.model_name_str in ["tasksource", "tasksource_v1", "bart-large"]:
            self.model.load_state_dict(new_model.state_dict())
            print("Resetting model of type", self.model_name_str)
        else:
            raise ValueError(f"Reset inplace operation is not implemented for model {self.model_name_str}")

    def _compute_raw_scores(self, sentence_pairs: List[Tuple[str, str]], show_progress=True) -> np.array:
        """ sentence pairs: List of evidence, claim pairs. """
        evidences = list([pair[0] for pair in sentence_pairs])
        claims = list([pair[1] for pair in sentence_pairs])

        if self.model_name_str in ['claude3-haiku', 'claude3-sonnet', 'llama3']:
            ret_list = []
            for b in wrap_tqdm(range(0, len(evidences), self.batch_size), show_progress):
                ret_list.append(get_scores(evidences[b:b+self.batch_size], claims[b:b+self.batch_size],
                                           self.model_id, system_prompt = MINICHECK_PROMPT, temperature=0.5,
                                           num_tries = 5, zero_one_prompt=True))
            return np.concatenate([s["scores"] for s in ret_list], axis=0)
        elif self.model_name_str in  ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
            return self.model.predict(sentence_pairs)
        elif self.model_name_str == "guardrails":
            return self.model.predict(sentence_pairs)
        elif self.model_name_str in ["vectara_v1", "nli-deberta-v3-base"]:
            res = self.model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False, batch_size=self.batch_size)
            return res
        elif self.model_name_str == "vectara_v2":
            tokenizer = self.model.tokenzier #sic!! (wrongly spelled in the package)
            score_list = []
            for b in wrap_tqdm(range(0, len(evidences), self.batch_size), show_progress):
                pair_dict = [{'text1': pair[0], 'text2': pair[1]} for pair in sentence_pairs[b:b+self.batch_size]]
                inputs = tokenizer(
                    [self.model.prompt.format(**pair) for pair in pair_dict], return_tensors='pt', padding=True)
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k]=inputs[k].to(self.device)
                self.model.t5.eval()
                with torch.no_grad():
                    outputs = self.model.t5(**inputs)
                logits = outputs.logits
                logits = logits[:, 0, :]  # tok_cls
                transformed_probs = torch.softmax(logits, dim=-1)
                raw_scores = transformed_probs[:, 1]  # the probability of class 1
                score_list.append(raw_scores)
            return torch.cat(score_list, dim=0).cpu().numpy()
        elif self.model_name_str in ['tasksource', 'tasksource_v1', 'bart-large', 'flan-t5-base']:
            if "split_sentences" in self.kvargs and self.kvargs['split_sentences'] == True:
                use_sentence_pairs = []
                num_sentences = []
                for ev, cl in sentence_pairs:
                    sentence_lvl_pairs = [[ev, s] for s in split_sentences(cl)]
                    num_sentences.append(len(sentence_lvl_pairs))
                    use_sentence_pairs += sentence_lvl_pairs
            else:
                use_sentence_pairs = sentence_pairs
            #print(len(num_sentences), num_sentences)
            ## Inference
            with torch.no_grad():
                x = self.tokenizer(use_sentence_pairs, return_tensors='pt', padding=True, truncation=True)
                batch_sz = self.batch_size
                scores_list= []
                for b in wrap_tqdm(range(0, len(x["input_ids"]), batch_sz), show_progress):
                    output = self.model(input_ids=x["input_ids"][b:b+batch_sz].to(self.device),
                                        attention_mask=x["attention_mask"][b:b+batch_sz].to(self.device))
                    scores_list.append(torch.softmax(output.logits, dim=-1)[:, 1])
            scores_list = torch.cat(scores_list)
            ## Aggregate sentence level
            if "split_sentences" in self.kvargs and self.kvargs['split_sentences'] == True:
                total_items = 0
                final_scores = []
                for num_split in num_sentences:
                    final_scores.append(torch.min(scores_list[total_items:total_items+num_split]).item())
                    total_items += num_split
                return np.array(final_scores)
            else:
                return scores_list.cpu().numpy()

        elif self.model_name_str == "alignscore-large" or self.model_name_str == "alignscore":
            if show_progress is False:
                with suppress():
                    scores = self.model.score(contexts=evidences, claims=claims)
            else:
                scores = self.model.score(contexts=evidences, claims=claims)
            return np.array(scores)
        elif "minicheck" in self.model_name_str:
            _, raw_prob, _, _ = self.model.score(docs=np.array(evidences), claims=np.array(claims))
            return np.array(raw_prob)
        elif "ensemble" in self.model_name_str:
            scores_list = []
            for unique_model in self.model:
                scores_list.append(unique_model.compute_scores(sentence_pairs))
            scores_list = np.stack(scores_list, axis=0)
            if self.model_name_str == "ensemble-mean":
                return np.mean(scores_list, axis=0)
            else: # median
                return np.median(scores_list, axis=0)


