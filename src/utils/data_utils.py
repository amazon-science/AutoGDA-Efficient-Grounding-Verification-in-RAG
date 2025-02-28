import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from .constants import NLI_LABELS, _BUCKET_NAME, _BUCKET_ROOTPATH
from src.utils.s3 import get_json_file_local_or_s3
from .s3 import read_csv_from_s3, read_jsonl_from_s3
import os
import random
import json
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import InputExample

class AnnotatedTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_id: str):
        assert "id" in df.columns
        assert "query" in df.columns
        assert "claim" in df.columns
        assert "evidence" in df.columns
        assert "label" in df.columns
        assert "label_binary" in df.columns

        self.data_id = data_id
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        dataset_id = str(row["id"])

        query = row["query"]
        group = str(row["group"])
        claim = row["claim"]
        citations = row["evidence"]
        if type(citations) is not str and np.isnan(citations):
            citations = "no relevant information found"
        label = str(row["label"])
        ret_dict = {
            "id": dataset_id,
            "query": query,
            "evidence": citations,
            "claim": claim,
            "group": group,
            "label": label,
            "label_binary": row["label_binary"],
        }
        if "label_cert" in row:
           ret_dict["label_cert"] = row["label_cert"]
        return ret_dict

    def get_dataset_identifier(self) -> str:
        return self.data_id


def _perform_query_integration(df_in, mode=None):
    """ Integrate the query into the evidence for the data_frame (useful for QA datasets).
        Permissible modes are: "prepend", query will be prepended, "queryonly", query will be used as evidence.
        This is useful for some datasets, where the query cell already contains the evidence.
        DOES NOT CHANGE THE ORIGINAL DF.
    """
    #print(mode)
    if mode is None:
        return df_in
    if mode=="prepend":
        df_out = df_in.copy()
        df_out["evidence"] = np.array(["You are given the question: " + r["query"] + " Here is some information related to the question: " + r["evidence"] for _, r in df_in.iterrows()])
    elif mode=="queryonly":
        df_out = df_in.copy()
        df_out["evidence"] = df_out["query"]
    else:
        raise ValueError("mode must be either 'prepend' or 'queryonly'")
    return df_out

def get_num_tokens(model, evidence: str, claim: str):
    texts = [[evidence, claim]]
    tokenized = model.tokenizer(
        texts,  return_tensors="np"
    )
    num_tokens = len(tokenized['input_ids'][0])
    return num_tokens


def filter_dataframe_by_length(dataset_df, filter_model_str, length_limit):
    """ Tokenize claim and evidence and filter out samples that have too many tokens. """
    if filter_model_str in ["cross-encoder/nli-deberta-v3-base", "vectara/hallucination_evaluation_model",
                            "allenai/longformer-base-4096"]:
        model = CrossEncoder(filter_model_str, num_labels=1)
        lengths = np.array([get_num_tokens(model, r[1]["claim"], r[1]["evidence"]) for r in dataset_df.iterrows()])
        dataset_df = dataset_df[lengths <= model.tokenizer.model_max_length]
    else:
        tokenizer = AutoTokenizer.from_pretrained(filter_model_str)
        lengths = np.array([tokenizer.encode(r["evidence"], r["claim"], return_tensors='pt').size(1) for i, r in
                            dataset_df.iterrows()])
        dataset_df = dataset_df[lengths <= length_limit]
    return dataset_df

def get_summedit_dataset(
    subset: str = "full", nli_labels: bool = False, scores_df: pd.DataFrame=None, stratified=True,
    filter_length=False, filter_model_str = "tasksource/deberta-base-long-nli", length_limit=1280
) -> Dataset:
    """ :param: filter_length: Filter out all inputs that exceed the context length of the model specified in filter_model_str.
        :param: stratified: If true, split across evidences, such that some evidences only appear in the train set.
    """
    data_id = f"Salesforce/summedits/{subset}"
    dataset = load_dataset("Salesforce/summedits")
    dataset_full = dataset["train"]

    if nli_labels:
        data_id = f"Salesforce/summedits-nli/{subset}"

    N = len(dataset_full)
    train_perc = 0.8
    N_train = int(N * train_perc)

    IND = list(range(0, N))
    random.seed(123)
    random.shuffle(IND)
    if not stratified:
        if subset == "train":
            i = IND[:N_train]
            dataset_sub = torch.utils.data.Subset(dataset_full, i)
        elif subset == "test":
            i = IND[N_train:]
            dataset_sub = torch.utils.data.Subset(dataset_full, i)
        else:
            dataset_sub = dataset_full
    else: # stratified: split by claims.
        dataset = pd.DataFrame(dataset_full)
        all_docs = dataset["doc"].unique()
        use_list_train = []
        use_list_test = []
        for doc_cnt, doc in enumerate(all_docs):
            doc_indices = dataset[dataset["doc"] == doc].index.values
            if doc_cnt % 5 == 1:
                use_list_test.append(doc_indices)
            else:
                use_list_train.append(doc_indices)
        if subset == "train" or subset =="val":
            idx_use = np.concatenate(use_list_train).tolist()
            #print(idx_use)
            #random.shuffle(idx_use)
            dataset_sub = torch.utils.data.Subset(dataset_full, idx_use)
        elif subset == "test":
            idx_use = np.concatenate(use_list_test).tolist()
            dataset_sub = torch.utils.data.Subset(dataset_full, idx_use)
        else:
            dataset_sub = dataset_full


    data_loader = DataLoader(dataset_sub)

    data_dict = {
        "id": [],
        "query": [],
        "claim": [],
        "evidence": [],
        "group": [],
        "label": [],
        "label_binary": [],
    }
    for batch in data_loader:
        label = int(batch["label"][0])
        label_binary = label
        if nli_labels and label == 0:
            edits = [e[0] for e in batch["edit_types"]]
            if len(edits) == 0:
                # Skip this sample
                continue
            else:
                # Get NLI label
                sample_edits = set(edits)
                neutral_edits = set(["hallucinated_fact_insertion"])
                contradiction_edits = set(
                    [
                        "entity_modification",
                        "negation_insertion_removal",
                        "antonym_swap",
                    ]
                )

                is_neutral = len(sample_edits.intersection(neutral_edits)) > 0
                is_contradiction = (
                    len(sample_edits.intersection(contradiction_edits)) > 0
                )
                # ---------------------------------------------------------------- #
                # Assign NLI labels.
                # NLI_LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}
                # ---------------------------------------------------------------- #
                label = 2 if is_neutral else 0

        data_dict["id"].append(str(batch["id"][0]))
        data_dict["query"].append("Paraphrase.")
        data_dict["claim"].append(batch["summary"][0])
        data_dict["evidence"].append(batch["doc"][0])
        data_dict["group"].append(batch["domain"][0])
        data_dict["label"].append(label)
        data_dict["label_binary"].append(label_binary)
    dataset_df = pd.DataFrame(data_dict)

    if filter_length:
        dataset_df = filter_dataframe_by_length(dataset_df, filter_model_str, length_limit)

    if subset == "val":
        dataset_df = dataset_df[np.arange(len(dataset_df)) % 5 == 1]
        if len(dataset_df)> 120:
            dataset_df = dataset_df.sample(n=60, random_state=42)
    elif subset =="train":
        dataset_df = dataset_df[np.arange(len(dataset_df)) % 5 != 1]
    if scores_df is not None:
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index

    return AnnotatedTextDataset(dataset_df, data_id=data_id)

def get_summedit_group_dataset(group: str,
                                subset: str = "full",
                                scores_df: pd.DataFrame = None,
                                stratified=True, filter_length=False,
                                filter_model_str = "tasksource/deberta-base-long-nli", length_limit=1280) -> Dataset:
    dataset_df = get_summedit_dataset(subset, stratified=stratified, filter_length=filter_length,
                                      filter_model_str=filter_model_str, length_limit=length_limit).df
    if isinstance(group, str):
        dataset_df = dataset_df[dataset_df["group"] == group]
        data_id = f"summedits-{group}/{subset}"
    elif isinstance(group, list):
        dataset_df = dataset_df[dataset_df["group"].isin(group)]
        data_id = f"summedits-{'_'.join(group)}/{subset}"
    elif group ==None:
        data_id = f"summedits/{subset}"
    if scores_df is not None:
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index

    return AnnotatedTextDataset(dataset_df, data_id=data_id)

def load_ragtruth(split="train", cols=None):
    local_response_path = "data/ragtruth-data/dataset/response.jsonl"
    local_source_path = "data/ragtruth-data/dataset/source_info.jsonl"

    df_response = pd.DataFrame(get_json_file_local_or_s3(f"{local_response_path}", local_response_path, _BUCKET_ROOTPATH, bucket=_BUCKET_NAME, lines=True))
    df_source = pd.DataFrame(get_json_file_local_or_s3(f"{local_source_path}", local_source_path, _BUCKET_ROOTPATH, bucket=_BUCKET_NAME, lines=True))


    df = df_response.merge(df_source, on="source_id", how='left')
    ## introduce val split
    if split in ["val", "train"]:
        evidence_list = list(df[df['split'] == "train"]["source_id"].unique())
        evidence_val = evidence_list[::4]
        df.loc[df["source_id"].isin(evidence_val), "split"] = "val"
    return df[(df['split'] == split) & (df['quality'] == 'good')][df.columns if cols is None else cols]

def get_ragtruth_dataset(split="train", group=None, filter_length=False, task="Summary",
                         filter_model_str = "tasksource/deberta-base-long-nli", length_limit=1280) -> Dataset:
    """ Task: "Summary", "QA", or None. Pass none to select all tasks.
        Splits: train, test, val.
    """
    if group is not None:
        data_id = f"ragtruth-{group}/{split}"
    else:
        data_id = f"ragtruth/{split}"
    df = load_ragtruth(split=split, cols=['model', 'source_id', 'labels', 'response', 'source_info', 'prompt', 'task_type'])

    ## Tasks: Drop data 2 text by default
    df = df[df["task_type"] != 'Data2txt']
    if task is not None:
        df = df[df["task_type"] == task]

    if group is not None: ## select only specific models.
        df = df[df["model"] == group]

    ###################
    # LABEL TYPE INFORMATION
    ###################
    priority_order = {
        'Evident Conflict': 1,
        'Subtle Conflict': 2,
        'Evident Baseless Info': 3,
        'Subtle Baseless Info': 4,
        'Other': 5
    }
    def keep_highest_priority_label(labels):
        if not labels:
            return 'entailment'

        highest_priority_label = min(labels, key=lambda x: priority_order.get(x['label_type'], 5))
        return highest_priority_label['label_type']

    df['label'] = df['labels'].apply(keep_highest_priority_label)
    # label contains either entailment or a key from priority order.

    # remove rows where final_label is 'other'
    df = df[df['label'] != 'Other']

    mapping_type = {
        'EVIDENT CONFLICT': 'CONTRADICTION',
        'SUBTLE CONFLICT': 'CONTRADICTION',
        'EVIDENT BASELESS INFO': 'NEUTRAL',
        'SUBTLE BASELESS INFO': 'NEUTRAL',
        'ENTAILMENT': 'ENTAILMENT'
    }

    df['label'] = df['label'].apply(lambda s: mapping_type[s.upper()])
    # ----------- #
    # Add binary label
    # ----------- #
    df['label_binary'] = (df['label'] == 'ENTAILMENT')

    # OTHER INFO
    # Add Number of Labels
    df['num_labels'] = df['labels'].apply(lambda s: len(s))


    # Modify source info for QA:
    # Use only documents as evidence
    df['source_info'] = df['source_info'].apply(lambda s: s if isinstance(s, str) else s["passages"])
    df.rename(columns={
        'source_id': 'id',
        'prompt': 'query',
        'response': 'claim',
        'source_info': 'evidence',
        'model': 'group'},
        inplace=True)

    # remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()]
    # KEEP FINAL COLUMNS
    selected_columns = ['id', 'query', 'claim', 'evidence', 'label', 'label_binary', 'group', 'num_labels']
    df = df[selected_columns]
    df = _perform_query_integration(df, mode="queryonly" if task=="QA" else None)
    if filter_length:
        df = filter_dataframe_by_length(df, filter_model_str, length_limit)

    if split == "val":
        val_set_size = get_validation_evidence_size("ragtruth", task)
        all_evlist = list(df.evidence.unique())
        df = df[df.evidence.isin(all_evlist[:val_set_size])]
    return AnnotatedTextDataset(df, data_id)

def get_validation_evidence_size(base_dataset, group):
    """ Get the number of evidences to use in a reasonable validation dataset (eg. ca. 100 labeled examples in total). """
    dict_vals = {("ragtruth", "QA"): 24, ("ragtruth", "Summary"): 24, ("summedits", "all"): 120,
                 ("lfqa-veri", "all"): 50}
    if isinstance(group, str):
        group = [group]
    group_sz = []
    for g in group:
        if (base_dataset, g) in dict_vals:
            group_sz.append(dict_vals[(base_dataset, g)])
        else:
            group_sz.append(None)
    if None in group_sz:
        return None
    else:
        return sum(group_sz)

def get_binary_label(annotation):
    numeric_dict = {"supported": 1, "partially": 0, "not_supported": -1}
    all_entail = True
    for item in annotation["annotations"]:
        res_annotations = np.array([numeric_dict[k] for k in item["labels"]])
        if np.sum(res_annotations) <= -2: ## Has one sentence with substantial non-entailment
            return 0
        if np.sum(res_annotations) < 2: ## There is at least disagreement
            all_entail = False
    if all_entail:
        return 1
    else:
        return -1 # Undecisive, should be filtered

def get_lfqa_verification(split="train", group=None, filter_model_str = "facebook/bart-large-mnli", length_limit=1024, filter_length=True):
    doc_annotation_list = [("annotations-gpt3_wdoc.json", "docs-webgpt.json"), ("annotations-alpaca_wdoc.json", "docs-webgpt.json"), ("annotations-gpt3_whudoc.json", "docs-human.json"), ("annotations-webgpt.json", "docs-webgpt.json")]

    df_dict_list = []
    for ann_file, doc_file in doc_annotation_list:
        annotations = get_json_file_local_or_s3(f"data/lfqa-verification/annotations/{ann_file}",
                                                ".", _BUCKET_ROOTPATH, bucket=_BUCKET_NAME)
        docs = get_json_file_local_or_s3(f"data/lfqa-verification/docs/{doc_file}",
                                                ".", _BUCKET_ROOTPATH, bucket=_BUCKET_NAME)
        #annotations = json.load(open(f"data/lfqa-verfication/annotations/{ann_file}"))
        #docs = json.load(open(f"data/lfqa-verfication/docs/{doc_file}"))
        for ann in annotations:
            lbl = get_binary_label(ann)
            if lbl == -1:
                continue
            line_dict = {}
            line_dict["id"] = ann["question_id"]
            line_dict["query"] = ann["question"]
            line_dict["claim"] = " ".join(ann["answers"])
            line_dict["claim"] = line_dict["claim"].strip(" ")
            if line_dict["claim"].endswith("</s>"):  # remove eos
                line_dict["claim"] = line_dict["claim"][:-4]
            line_dict["label"] = "ENTAILMENT" if lbl==1 else "CONTRADICTION"
            line_dict["label_binary"] = lbl
            line_dict["group"] = ann_file
            relevant_docs = docs[ann["question_id"]]
            if relevant_docs["question_id"] != ann["question_id"]:
                print("Question mismatch: ", ann_file, ann["question_id"], relevant_docs["question_id"])
            passage_list = []
            for idx, item in enumerate(relevant_docs["docs"]):
                passage_list.append(item['text'])
            line_dict["evidence"] = " ".join(passage_list)
            line_dict["evidence"] = line_dict["evidence"].strip(" ")
            df_dict_list.append(line_dict)
    lfqa_very = pd.DataFrame(df_dict_list)
    lfqa_very = lfqa_very.sample(frac=1, random_state=1)
    ev_list = list(lfqa_very.evidence.unique())
    if split == "train":
        lfqa_very = lfqa_very[lfqa_very["evidence"].isin(ev_list[:85])]
    if split == "val":
        lfqa_very = lfqa_very[lfqa_very["evidence"].isin(ev_list[85:110])]
    if split == "test":
        lfqa_very = lfqa_very[lfqa_very["evidence"].isin(ev_list[110:])]
    lfqa_very = _perform_query_integration(lfqa_very, mode="prepend")
    if filter_length:
        lfqa_very = filter_dataframe_by_length(lfqa_very, filter_model_str, length_limit)
    return AnnotatedTextDataset(lfqa_very, data_id=f"LFQA-Verify/{split}")

def prepare_samples(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str, do_val=True):

    train_samples, val_samples, test_samples = [], [], []

    train_df = train_df.sample(frac=1, random_state=0)
    if do_val:
        N_train = int(len(train_df) * 0.8)
    else:
        N_train = len(train_df) ## all for train

    train2_df = train_df.iloc[:N_train, :]
    val_df = train_df.iloc[N_train:, :]

    # Use 80% of the train data for training
    for e, c, l in zip(train2_df["evidence"], train2_df["claim"], train2_df[label_col]):
        train_samples.append(InputExample(texts=[e, c], label=int(l)))

    # Hold 20% of the train data for validation
    for e, c, l in zip(val_df["evidence"], val_df["claim"], val_df[label_col]):
        val_samples.append(InputExample(texts=[e, c], label=int(l)))

    # TEST DATASET
    for e, c, l in zip(test_df["evidence"], test_df["claim"], test_df[label_col]):
        test_samples.append(InputExample(texts=[e, c], label=int(l)))

    return train_samples, val_samples, test_samples