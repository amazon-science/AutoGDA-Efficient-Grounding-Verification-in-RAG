import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from .constants import NLI_LABELS, _SD_BUCKET_NAME, _BUCKET_NAME
from .s3 import read_csv_from_s3, read_jsonl_from_s3
import os
import random
import json
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModel
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


def perform_query_integration(df_in, mode=None):
    """ Integrate the query into the evidence for the data_frame (useful for QA datasets).
        Permissible modes are: "prepend", query will be prepended, "queryonly", query will be used as evidence.
        This is useful for some datasets, where the query cell already contains the evidence.
        DOES NOT CHANGE THE ORIGINAL DF.
    """
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

def get_vitaminc_dataset_train(scores_df: pd.DataFrame = None) -> Dataset:
    return get_vitaminc_dataset(train_or_test="train", scores_df=scores_df)


def get_vitaminc_dataset_test(scores_df: pd.DataFrame = None) -> Dataset:
    return get_vitaminc_dataset(train_or_test="test", scores_df=scores_df)


def get_vitaminc_dataset(
    train_or_test="test", scores_df: pd.DataFrame = None
) -> Dataset:
    data_id = f"tals/vitaminc/{train_or_test}"
    dataset = load_dataset("tals/vitaminc")

    data_loader = DataLoader(dataset[train_or_test])
    data_dict = {
        "id": [],
        "query": [],
        "claim": [],
        "evidence": [],
        "group": [],
        "label": [],
    }
    for batch in data_loader:
        data_dict["id"].append(str(batch["unique_id"][0]))
        data_dict["query"].append(f"Tell me a fact about {batch['page'][0]}")
        data_dict["claim"].append(batch["claim"][0])
        data_dict["evidence"].append(batch["evidence"][0])
        data_dict["group"].append(batch["page"][0])
        data_dict["label"].append(batch["label"][0])

    dataset_df = pd.DataFrame(data_dict)

    MAPPER = {
        "SUPPORTS": "ENTAILMENT",
        "REFUTES": "CONTRADICTION",
        "NOT ENOUGH INFO": "NEUTRAL",
    }
    dataset_df["label"] = dataset_df["label"].apply(lambda s: MAPPER[s])
    dataset_df["label"] = (
        dataset_df["label"].apply(lambda s: NLI_LABELS[s.lower()]).astype(int)
    )
    dataset_df["label_binary"] = (dataset_df["label"] == 1).astype(int)

    if scores_df is not None:
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index

    return AnnotatedTextDataset(dataset_df, data_id=data_id)


def get_paws_dataset(scores_df: pd.DataFrame = None, subset="test") -> Dataset:
    data_id = f"paws_labeled_final/{subset}"
    dataset = load_dataset("paws", "labeled_final")[subset]
    length = [50, 72, 94, 116, 138, 160, 182, 250]

    def get_group_len(s):
        l = len(s)
        g = len(length)
        for i in range(len(length) - 1):
            if length[i + 1] > l:
                g = i
                break
        if g == len(length):
            return f"evidence.length > {length[-1]}"
        else:
            return f"evidence.length < {length[g+1]}"

    data_loader = DataLoader(dataset)
    data_dict = {
        "id": [],
        "query": [],
        "claim": [],
        "evidence": [],
        "group": [],
        "label": [],
    }
    for batch in data_loader:
        data_dict["id"].append(int(batch["id"][0]))
        data_dict["query"].append(f"paraphrase")
        data_dict["claim"].append(batch["sentence1"][0])
        data_dict["evidence"].append(batch["sentence2"][0])
        data_dict["group"].append(get_group_len(batch["sentence2"][0]))
        data_dict["label"].append(int(batch["label"][0]))

    data_dict["label_binary"] = data_dict["label"]
    dataset_df = pd.DataFrame(data_dict)

    if scores_df is not None:
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index
    return AnnotatedTextDataset(dataset_df, data_id=data_id)


# nyu-mll/multi_nli
def get_nli_dataset(
    subset: str = "validation_matched", scores_df: pd.DataFrame = None
) -> Dataset:
    data_id = f"nyu-mll/multi_nli/{subset}"
    dataset = load_dataset(f"nyu-mll/multi_nli")

    data_loader = DataLoader(dataset[subset])
    data_dict = {
        "id": [],
        "query": [],
        "claim": [],
        "evidence": [],
        "group": [],
        "label": [],
        "label_binary": [],
    }

    idtolabel = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for batch in data_loader:
        data_dict["id"].append(str(batch["pairID"][0]))
        data_dict["query"].append(str(batch["promptID"]))
        data_dict["claim"].append(batch["hypothesis"][0])
        data_dict["evidence"].append(batch["premise"][0])
        data_dict["group"].append(batch["genre"][0])
        label_str = idtolabel[int(batch["label"][0])]
        data_dict["label"].append(NLI_LABELS[label_str])
        data_dict["label_binary"].append(label_str == "entailment")

    dataset_df = pd.DataFrame(data_dict)

    if scores_df is not None:
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index

    return AnnotatedTextDataset(dataset_df, data_id=data_id)

def get_num_tokens(model, evidence: str, claim: str):
    texts = [[evidence, claim]]
    tokenized = model.tokenizer(
        texts,  return_tensors="np"
    )
    num_tokens = len(tokenized['input_ids'][0])
    return num_tokens

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
        if subset == "train":
            idx_use = np.concatenate(use_list_train).tolist()
            #print(idx_use)
            random.shuffle(idx_use)
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
        if filter_model_str in ["cross-encoder/nli-deberta-v3-base", "vectara/hallucination_evaluation_model", "allenai/longformer-base-4096"]:
            model = CrossEncoder(filter_model_str, num_labels=1)
            lengths = np.array([get_num_tokens(model, r[1]["claim"], r[1]["evidence"]) for r in dataset_df.iterrows()])
            dataset_df = dataset_df[lengths <= model.tokenizer.model_max_length]
        else:
            tokenizer = AutoTokenizer.from_pretrained(filter_model_str)
            lengths = np.array([tokenizer.encode(r["evidence"], r["claim"], return_tensors='pt').size(1) for i, r in dataset_df.iterrows()])
            dataset_df = dataset_df[lengths <= length_limit]
    if scores_df is not None:
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index

    return AnnotatedTextDataset(dataset_df, data_id=data_id)


def get_fever_train(scores_df: pd.DataFrame = None) -> Dataset:
    dataset_df = read_csv_from_s3(
        "harpo-hallucination-detection", f"datasets/fever/train/dataset.csv"
    )
    dataset_df['group'] = dataset_df['group'].apply(lambda x: "No group")
    dataset_df = dataset_df.dropna()
    data_id = "fever/train"

    if scores_df is not None:
        # scores_results = pd.read_csv(results_path, index_col=0)
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index

    return AnnotatedTextDataset(dataset_df, data_id=data_id)


def get_fever_test(scores_df: pd.DataFrame = None) -> Dataset:
    dataset_df = read_csv_from_s3(
        "harpo-hallucination-detection", f"datasets/fever/test/dataset.csv"
    )
    dataset_df['group'] = dataset_df['group'].apply(lambda x: "No group")
    dataset_df = dataset_df.dropna()
    data_id = "fever/test"

    if scores_df is not None:
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index

    return AnnotatedTextDataset(dataset_df, data_id=data_id)


def get_claude_sonnet_dataset(
    scores_df: pd.DataFrame = None,
    fnms=[
        "claude3-sonnet-nli.csv",
        "claude3-sonnet-nli-1.csv",
        "claude3-sonnet-nli-2.csv",
    ],
) -> Dataset:
    data_id = "claude-sonnet"
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    dfs = [read_csv_from_s3('synthetic-nli-data', fnm) for fnm in fnms]
    df = pd.concat(dfs)
    df["id"] = "claude-sonnet"
    df["group"] = df["query"] = ""
    df.rename({"premise": "evidence", "hypothesis": "claim"}, inplace=True, axis=1)
    df["label"] = [label2int[el] for el in df["label"]]
    df["label_binary"] = [int(el==1) for el in df["label"]]
    df["premise"] = df["evidence"].fillna("No supporting evidence")
    df["hypothesis"] = df["claim"].fillna("No response")

    if scores_df is not None:
        df = scores_df.set_index("id").join(df.set_index("id"))
        df["id"] = df.index
    return AnnotatedTextDataset(df, data_id)


def get_aza_sd(
    scores_df: pd.DataFrame = None,
    fnms=["synthetic-data-sets/synthetic_aza_sonnet.csv"],
) -> Dataset:
    data_id = "claude-sonnet"
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    dfs = [read_csv_from_s3("synthetic-calibration-data-new", fnm) for fnm in fnms]
    df = pd.concat(dfs)
    df["id"] = "claude-sonnet"
    df["group"] = df["query"] = ""
    df.rename({"premise": "evidence", "hypothesis": "claim"}, inplace=True, axis=1)
    df["label"] = [label2int[el] for el in df["label"]]
    print(df.columns)
    df["premise"] = df["evidence"].fillna("No supporting evidence")
    df["hypothesis"] = df["claim"].fillna("No response")

    if scores_df is not None:
        # scores_results = pd.read_csv(results_path, index_col=0)
        df = scores_df.set_index("id").join(df.set_index("id"))
        df["id"] = df.index

    os.makedirs("temp", exist_ok=True)
    local_path = "temp/aza.csv"
    print(f"Save local", local_path)
    df.to_csv(local_path)
    return AnnotatedTextDataset(df, data_id)


label_map = {
    "FACTUAL": "FACTUAL",
    "CONTRADICTION": "HALLUCINATION",
    "NEUTRAL": "HALLUCINATION",
    "CONTRADICTION-NEUTRAL": "HALLUCINATION",
}


def get_plato_dataset_sentence(scores_df: pd.DataFrame = None) -> Dataset:
    """ Split claims in plato dataset into sentences.
     Posible lables are:
     FACTUAL: All subclaims in the claim sentence are supported by the evidence. There might be some minor details
            deviating from the evidence information, but these changes do not impact the main idea in the evidence document. You
            should only use the provided evidence to verify the claim and not your own knowledge.
    CONTRADICTION: At least one subclaim in the claim setence is contradicted by the evidence.
    NEUTRAL: At least one subclaim in the claim sentence is not coming from the evidence.
    CONTRADICTION+NEUTRAL: This means that the claim sentence contains both CONTRADICTION and NEUTRAL content.
    INTRO: Use this label for introductory sentences. Introductory sentences does not make any claims that need to be
            verified against the evidence. For example, a sentence that is setting up the context for the subsequent claims.
     """

    # plato_data = get_plato_dataset()
    data_id = 'plato_sentence'
    model_name = 'claude3-sonnet'
    save_path = f'datasets/{data_id}/{model_name}/dataset.csv'

    # {"contradiction": 0, "entailment": 1, "neutral": 2
    df = read_csv_from_s3('harpo-hallucination-detection', save_path)

    # ----------- #
    # Drop INTRO and CONTRADICTION+NEUTRAL labels
    # ----------- #
    df = df.drop(df[df['label'] == 'INTRO'].index)
    df = df.drop(df[df['label'] == 'CONTRADICTION+NEUTRAL'].index)

    mapping = {'FACTUAL': 'ENTAILMENT', 'ENTAILMENT': 'ENTAILMENT',
               'CONTRADICTION': 'CONTRADICTION', 'NEUTRAL': 'NEUTRAL' }
    df['label'] = df['label'].apply(lambda s: mapping[s.upper()] )
    df['label'] = df['label'].apply(lambda s: NLI_LABELS[s.lower()] ).astype(int)

    # ----------- #
    # Add binary label
    # ----------- #
    df['label_binary'] = df['label'] == 1


    if scores_df is not None:
        if 'reasoning' in df.columns and 'reasoning' in scores_df.columns :
            df = df.drop(columns=['reasoning'], index=1)
        df = scores_df.set_index('id').join(df.set_index('id'))
        df['id'] = df.index

    df = df.dropna()
    df['label'] = df['label'].astype(int)
    return AnnotatedTextDataset(df, data_id)




def get_summedit_sync_data(scores_df: pd.DataFrame = None) -> Dataset:
    data_id = f'summedits-sync'
    groups = ['news', 'podcast', 'billsum', 'samsum', 'sales_call', 'sales_email', 'shakespeare', 'scitldr','qmsumm', 'ectsum']
    df_list = []
    for g in groups:
        path = f'sync_data/summedits-sync-{g}.csv'
        print(f'Loading {path}')
        df = read_csv_from_s3(_BUCKET_NAME, path)
        df['label'] = df['label'].apply(lambda s: NLI_LABELS[s.lower()] )
        df['label_binary'] = df['label_binary'].astype(int)
        df['group'] = g
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    df['id'] = np.arange(len(df))
    print(df.head())
    print()
    if scores_df is not None:
        if 'reasoning' in df.columns and 'reasoning' in scores_df.columns :
            df = df.drop(columns=['reasoning'], index=1)
        df = scores_df.set_index('id').join(df.set_index('id'))
        df['id'] = df.index

    df = df.dropna()
    return AnnotatedTextDataset(df, data_id)

def get_summedit_sync_data_group(group: str, scores_df: pd.DataFrame = None, tag=None) -> Dataset:
    dataset_name = "Salesforce/summedits/full"
    model_name = "claude3-sonnet"
    if tag is None:
        data_id = f"summedits-sync-{group}"
    else:
        data_id = f"summedits-sync-{group}-{tag}"
    path = f"sync_data/{data_id}.csv"
    df = read_csv_from_s3('harpo-hallucination-detection', path)

    mapping = {
        "factual": "entailment",
        "entailment": "entailment",
        "contradiction": "contradiction",
        "neutral": "neutral",
    }
    df["group"] = group

    df["label"] = df["label"].apply(lambda s: mapping[s.lower()])
    df["label"] = df["label"].apply(lambda s: NLI_LABELS[s.lower()])
    df["label_binary"] = (df["label"] == 1)
    df = df.dropna()
    return AnnotatedTextDataset(df, data_id=data_id)

def get_summedits_llm_labeled_group_data(group: str, subset: str = "train"):
    df = pd.read_csv(f'src/llm_entailment_scores/llm_labeled_Salesforce-summedits-nli-{subset}.csv')
    df = df[df.group==group]
    df["label_binary"] = df["llm_scores"].astype(int)
    return AnnotatedTextDataset(df, data_id=f"llmlabeled-summedits-{group}-{subset}")

def get_aza_sync_train(scores_df: pd.DataFrame = None) -> Dataset:
    return get_aza_sync(split="train", scores_df=scores_df)


def get_aza_sync_test(scores_df: pd.DataFrame = None) -> Dataset:
    return get_aza_sync(split="test", scores_df=scores_df)

def get_summedit_group_dataset(group: str,
                                subset: str = "full",
                                scores_df: pd.DataFrame = None,
                                stratified=False, filter_length=False,
                                filter_model_str = "tasksource/deberta-base-long-nli", length_limit=1280) -> Dataset:
    data_id = f"summedits-{group}/{subset}"
    dataset_df = get_summedit_dataset(subset, stratified=stratified, filter_length=filter_length,
                                      filter_model_str=filter_model_str, length_limit=length_limit).df
    dataset_df = dataset_df[dataset_df["group"] == group]
    if scores_df is not None:
        dataset_df = scores_df.set_index("id").join(dataset_df.set_index("id"))
        dataset_df["id"] = dataset_df.index

    return AnnotatedTextDataset(dataset_df, data_id=data_id)

def load_ragtruth(split="train", cols=None):
    local_response_path = "data/raw/ragtruth-data/dataset/response.jsonl"
    local_source_path = "data/raw/ragtruth-data/dataset/source_info.jsonl"
    response_path = "dataset/response.jsonl"
    source_path = "dataset/source_info.jsonl"
    bucket = "ragtruth-data"

    def read_local_jsonl(file_path: str) -> pd.DataFrame:
        with open(file_path, 'r') as f:
            return list([json.loads(line) for line in f])

    try:
        #logger.info("Attempting to load data from local files...")
        df_response = pd.DataFrame(read_local_jsonl(local_response_path))
        df_source = pd.DataFrame(read_local_jsonl(local_source_path))
        #logger.info("Successfully loaded data from local files.")
    except FileNotFoundError:
        print("Local files for RAGTruth not found. Falling back to S3...")
        try:
            df_response = pd.DataFrame(read_jsonl_from_s3(bucket, response_path))
            df_source = pd.DataFrame(read_jsonl_from_s3(bucket, source_path))
            #logger.info("Successfully loaded data from S3.")
        except Exception as e:
            #logger.error(f"Failed to load data from S3: {str(e)}")
            raise


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

    if filter_length:
        if filter_model_str in ["cross-encoder/nli-deberta-v3-base", "vectara/hallucination_evaluation_model", "allenai/longformer-base-4096"]:
            model = CrossEncoder(filter_model_str, num_labels=1)
            lengths = np.array([get_num_tokens(model, r[1]["claim"], r[1]["evidence"]) for r in df.iterrows()])
            df = df[lengths <= model.tokenizer.model_max_length]
        else:
            tokenizer = AutoTokenizer.from_pretrained(filter_model_str)
            lengths = np.array([tokenizer.encode(r["evidence"], r["claim"], return_tensors='pt').size(1) for i, r in df.iterrows()])
            print(np.max(lengths))
            df = df[lengths <= length_limit]

    return AnnotatedTextDataset(df, data_id)

def get_expertqa_data(group: str=None, subset: str = "train", file_path = "data/expertQA/r2_compiled_anon.jsonl",
                      share_train=0.5, filter_length=True, filter_model_str="tasksource/deberta-base-long-nli", length_limit=1280):
    with open(file_path, 'r') as f:
        dictlist = list([json.loads(line) for line in f])
    expert_qa_data = pd.DataFrame(dictlist)
    expert_qa_data["is_test"] = 1
    expert_qa_data = expert_qa_data.sample(frac=1, random_state=1)
    n_train = int(len(expert_qa_data) * share_train)
    expert_qa_data.iloc[:n_train, expert_qa_data.columns.get_loc('is_test')] = 0

    def unroll_claims(row):
        ret_df = pd.concat([pd.DataFrame(row["answers"][k]["claims"]).assign(model=k) for k in row["answers"].keys()],
                           axis=0, ignore_index=True)
        ret_df["model"] = row["metadata"]["field"]
        ret_df["query"] = row["question"]
        ret_df["is_test"] = row["is_test"]
        return ret_df

    all_claims = pd.concat([unroll_claims(row) for _, row in expert_qa_data.iterrows()], ignore_index=True)
    all_claims = all_claims[all_claims.support.isin(["Complete", "Incomplete"])]

    ## Filter only claims with textual evidence
    def filter_evidence(input_row):
        ret = ""
        for k in input_row:
            parts = k.split("\n")
            if len(parts) > 1:
                ret += (" ".join(parts[1:])).strip(" ")
        # Filter citations in square brackets, e.g. [3]
        return ret

    def filter_claim(ret):
        while "[" in ret:
            len_del = 1
            loc = ret.index('[')
            loc2 = ret.index(']', loc)
            if loc2 > 0 and loc2 - loc <= 3:
                len_del = loc2 - loc
            if loc > 0 and ret[loc - 1] == " ":
                loc = loc - 1
                len_del = len_del + 1
            ret = ret[:loc] + ret[loc + len_del + 1:]
        return ret
    all_claims["evidence_rev"] = all_claims["evidence"].apply(filter_evidence)
    all_claims.drop(["evidence", "reason_missing_support", "source_reliability", "revised_claim_string",
                     "informativeness", "worthiness", "correctness", "revised_claim", "revised_evidence", "reliability"], axis=1, inplace=True)
    all_claims = all_claims[all_claims["evidence_rev"] != ""]
    print("Filter claims.")
    all_claims["claim_string"] = all_claims["claim_string"].apply(filter_claim)
    all_claims["id"] = np.arange(len(all_claims))

    if group is not None:
        group_mappings = {"Healthcare": "Healthcare / Medicine", "Engineering": "Engineering and Technology", "VisualArts": "Visual Arts"}
        if group in group_mappings:
            group_use = group_mappings[group]
        else:
            group_use = group
        all_claims = all_claims[all_claims["model"] ==group_use]
    if subset is not None:
        all_claims = all_claims[all_claims["is_test"] == (0 if subset=="train" else 1)]

    all_claims['label'] = all_claims['support'].map({"Complete": "ENTAILMENT", "Incomplete": "NEUTRAL"})
    all_claims['label_binary'] = (all_claims['label']  == "ENTAILMENT")
    all_claims.rename(columns={
        'claim_string': 'claim',
        'prompt': 'query',
        'evidence_rev': 'evidence',
        'model': 'group'},
        inplace=True)
    data_id = "expertqa"
    if group is not None:
        data_id += ("-" + group)
    if subset is not None:
        data_id += ("/" + subset)

    if filter_length:
        if filter_model_str in ["cross-encoder/nli-deberta-v3-base", "vectara/hallucination_evaluation_model", "allenai/longformer-base-4096"]:
            model = CrossEncoder(filter_model_str, num_labels=1)
            lengths = np.array([get_num_tokens(model, r[1]["claim"], r[1]["evidence"]) for r in all_claims.iterrows()])
            all_claims = all_claims[lengths <= model.tokenizer.model_max_length]
        else:
            tokenizer = AutoTokenizer.from_pretrained(filter_model_str)
            lengths = np.array([tokenizer.encode(r["evidence"], r["claim"], return_tensors='pt').size(1) for i, r in all_claims.iterrows()])
            all_claims = all_claims[lengths <= length_limit]

    return AnnotatedTextDataset(all_claims, data_id=data_id)


def get_ragtruth_llm_labeled_data(group: str = None, subset: str = "train"):
    if subset == "train":
        df = pd.read_csv(f'llm_labeled_ragtruth-all-train.csv')
    else:
        df = pd.read_csv(f'llm_labeled_ragtruth-all-test.csv')
    if group is not None:
        df = df[df.group==group]
    df["label_binary"] = df["llm_scores"].astype(int)

    if group is not None:
        data_id = f"ragtruth-{group}/{subset}"
    else:
        data_id = f"ragtruth/{subset}"
    return AnnotatedTextDataset(df, data_id=data_id)

def get_ragtruth_sync_data(group: str = None, subset: str = "train", filter = None):
    if subset == "train":
        if filter is not None:
            df = pd.read_csv(f'sync_data/ragtruth-all-train-2000-{filter}.csv')
        else:
            df = pd.read_csv(f'sync_data/ragtruth-all-train.csv')
    else:
        df = pd.read_csv(f'sync_data/ragtruth-all-test.csv')
    if group is not None:
        df = df[df.group==group]

    if group is not None:
        data_id = f"ragtruth-{group}/{subset}"
    else:
        data_id = f"ragtruth/{subset}"
    if filter is not None:
        data_id += ("-" + filter)
    return AnnotatedTextDataset(df, data_id=data_id)


def get_aza_sync(split="train", scores_df: pd.DataFrame = None) -> Dataset:
    data_id = f"aza_sync/{split}"
    path = "sync_data/sync_aza_balanced.csv"
    bucket = "synthetic-calibration-data-new"
    df = read_csv_from_s3(bucket, path)
    df = df.rename(
        {"premise": "evidence", "hypothesis": "claim", "Unnamed: 0": "id"}, axis=1
    )
    df = df.sample(frac=1, random_state=0)
    df["group"] = "None"
    df["query"] = "Provide a claim relevant to the evidence."
    mapping = {
        "factual": "entailment",
        "entailment": "entailment",
        "contradiction": "contradiction",
        "neutral": "neutral",
    }
    df["label"] = df["label"].apply(lambda s: mapping[s.lower()])
    df["label"] = df["label"].apply(lambda s: NLI_LABELS[s.lower()])
    df["label_binary"] = (df["label"] == 1).astype(int)
    df = df.dropna()

    N_train = int(len(df) * 0.8)
    if split == "train":
        df = df[:N_train]
    else:
        df = df[N_train:]
    return AnnotatedTextDataset(df, data_id)


DATA_LOADER_FN = {
    "tals/vitaminc/train": get_vitaminc_dataset_train,
    "tals/vitaminc/test": get_vitaminc_dataset_test,
    "fever/train": get_fever_train,
    "fever/test": get_fever_test,
    "paws_labeled_final/train": lambda scores_df=None: get_paws_dataset(
        scores_df, subset="train"
    ),
    "paws_labeled_final/test": lambda scores_df=None: get_paws_dataset(
        scores_df, subset="test"
    ),
    "nyu-mll/multi_nli/validation_matched": get_nli_dataset,
    "Salesforce/summedits/train": lambda scores_df=None: get_summedit_dataset(
        subset="train", scores_df=scores_df
    ),
    "Salesforce/summedits/test": lambda scores_df=None: get_summedit_dataset(
        subset="test", scores_df=scores_df
    ),
    "Salesforce/summedits/full": lambda scores_df=None: get_summedit_dataset(
        subset="full", scores_df=scores_df
    ),
    "Salesforce/summedits-nli/train": lambda scores_df=None: get_summedit_dataset(
        subset="train", nli_labels=True, scores_df=scores_df
    ),
    "Salesforce/summedits-nli/test": lambda scores_df=None: get_summedit_dataset(
        subset="test", nli_labels=True, scores_df=scores_df
    ),
    "Salesforce/summedits-nli/full": lambda scores_df=None: get_summedit_dataset(
        subset="full", nli_labels=True, scores_df=scores_df
    ),
    "plato_sentence": get_plato_dataset_sentence,
    "claude-sonnet-sd": get_claude_sonnet_dataset,
    "claude-sonnet": get_claude_sonnet_dataset,
    "summedits-sync": get_summedit_sync_data,
    "aza_sync/train": get_aza_sync_train,
    "aza_sync/test": get_aza_sync_test,
}


DATA_LOADER_FN[f"summedits-news/train"] = lambda scores_df=None: get_summedit_group_dataset(group="news", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-news/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="news", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-news/test"] = lambda scores_df=None: get_summedit_group_dataset(group="news", subset="test", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-news/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="news", subset="train")

DATA_LOADER_FN[f"summedits-podcast/train"] = lambda scores_df=None: get_summedit_group_dataset(group="podcast", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-podcast/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="podcast", subset="train")
DATA_LOADER_FN[f"summedits-podcast/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="podcast", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-podcast/test"] = lambda scores_df=None: get_summedit_group_dataset(group="podcast", subset="test", scores_df=scores_df)

DATA_LOADER_FN[f"summedits-billsum/train"] = lambda scores_df=None: get_summedit_group_dataset(group="billsum", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-billsum/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="billsum", subset="train")
DATA_LOADER_FN[f"summedits-billsum/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="billsum", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-billsum/syncv2haiku"] = lambda scores_df=None: get_summedit_sync_data_group(group="billsumm", scores_df=scores_df, tag="v2haiku")
DATA_LOADER_FN[f"summedits-billsum/syncv2sonnet"] = lambda scores_df=None: get_summedit_sync_data_group(group="billsumm", scores_df=scores_df, tag="v2sonnet")
DATA_LOADER_FN[f"summedits-billsum/test"] = lambda scores_df=None: get_summedit_group_dataset(group="billsum", subset="test", scores_df=scores_df)

DATA_LOADER_FN[f"summedits-samsum/train"] = lambda scores_df=None: get_summedit_group_dataset(group="samsum", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-samsum/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="samsum", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-samsum/test"] = lambda scores_df=None: get_summedit_group_dataset(group="samsum", subset="test", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-samsum/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="samsum", subset="train")

DATA_LOADER_FN[f"summedits-sales_call/train"] = lambda scores_df=None: get_summedit_group_dataset(group="sales_call", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-sales_call/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="sales_call", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-sales_call/test"] = lambda scores_df=None: get_summedit_group_dataset(group="sales_call", subset="test", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-sales_call/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="sales_call", subset="train")

DATA_LOADER_FN[f"summedits-sales_email/train"] = lambda scores_df=None: get_summedit_group_dataset(group="sales_email", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-sales_email/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="sales_email", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-sales_email/test"] = lambda scores_df=None: get_summedit_group_dataset(group="sales_email", subset="test", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-sales_email/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="sales_email", subset="train")

DATA_LOADER_FN[f"summedits-shakespeare/train"] = lambda scores_df=None: get_summedit_group_dataset(group="shakespeare", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-shakespeare/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="shakespeare", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-shakespeare/test"] = lambda scores_df=None: get_summedit_group_dataset(group="shakespeare", subset="test", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-shakespeare/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="shakespeare", subset="train")

DATA_LOADER_FN[f"summedits-scitldr/train"] = lambda scores_df=None: get_summedit_group_dataset(group="scitldr", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-scitldr/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="scitldr", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-scitldr/test"] = lambda scores_df=None: get_summedit_group_dataset(group="scitldr", subset="test", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-scitldr/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="scitldr", subset="train")

DATA_LOADER_FN[f"summedits-qmsumm/train"] = lambda scores_df=None: get_summedit_group_dataset(group="qmsumm", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-qmsumm/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="qmsumm", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-qmsumm/test"] = lambda scores_df=None: get_summedit_group_dataset(group="qmsumm", subset="test", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-qmsumm/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="qmsumm", subset="train")

DATA_LOADER_FN[f"summedits-ectsum/train"] = lambda scores_df=None: get_summedit_group_dataset(group="ectsum", subset="train", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-ectsum/sync"] = lambda scores_df=None: get_summedit_sync_data_group(group="ectsum", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-ectsum/test"] = lambda scores_df=None: get_summedit_group_dataset(group="ectsum", subset="test", scores_df=scores_df)
DATA_LOADER_FN[f"summedits-ectsum/trainllm"] = lambda scores_df=None: get_summedits_llm_labeled_group_data(group="ectsum", subset="train")

##RAGTRUTH
DATA_LOADER_FN[f"ragtruth-gpt-4/train"] = lambda scores_df=None: get_ragtruth_dataset(split="train", group='gpt-4-0613', filter_length=True)
DATA_LOADER_FN[f"ragtruth-gpt-4/trainllm"] = lambda scores_df=None: get_ragtruth_llm_labeled_data(subset="train", group='gpt-4-0613')
DATA_LOADER_FN[f"ragtruth-gpt-4/test"] = lambda scores_df=None: get_ragtruth_dataset(split="test", group='gpt-4-0613', filter_length=True)

DATA_LOADER_FN[f"ragtruth-gpt-3.5/train"] = lambda scores_df=None: get_ragtruth_dataset(split="train", group='gpt-3.5-turbo-0613', filter_length=True)
DATA_LOADER_FN[f"ragtruth-gpt-3.5/trainllm"] = lambda scores_df=None: get_ragtruth_llm_labeled_data(subset="train", group='gpt-3.5-turbo-0613')
DATA_LOADER_FN[f"ragtruth-gpt-3.5/test"] = lambda scores_df=None: get_ragtruth_dataset(split="test", group='gpt-3.5-turbo-0613', filter_length=True)

DATA_LOADER_FN[f"ragtruth-mistral-7b/train"] = lambda scores_df=None: get_ragtruth_dataset(split="train", group='mistral-7B-instruct', filter_length=True)
DATA_LOADER_FN[f"ragtruth-mistral-7b/trainllm"] = lambda scores_df=None: get_ragtruth_llm_labeled_data(subset="train", group='mistral-7B-instruct')
DATA_LOADER_FN[f"ragtruth-mistral-7b/test"] = lambda scores_df=None: get_ragtruth_dataset(split="test", group='mistral-7B-instruct', filter_length=True)

DATA_LOADER_FN[f"ragtruth-llama-7b/train"] = lambda scores_df=None: get_ragtruth_dataset(split="train", group='llama-2-7b-chat', filter_length=True)
DATA_LOADER_FN[f"ragtruth-llama-7b/trainllm"] = lambda scores_df=None: get_ragtruth_llm_labeled_data(subset="train", group='llama-2-7b-chat')
DATA_LOADER_FN[f"ragtruth-llama-7b/test"] = lambda scores_df=None: get_ragtruth_dataset(split="test", group='llama-2-7b-chat', filter_length=True)

DATA_LOADER_FN[f"ragtruth-llama-13b/train"] = lambda scores_df=None: get_ragtruth_dataset(split="train", group='llama-2-13b-chat', filter_length=True)
DATA_LOADER_FN[f"ragtruth-llama-13b/trainllm"] = lambda scores_df=None: get_ragtruth_llm_labeled_data(subset="train", group='llama-2-13b-chat')
DATA_LOADER_FN[f"ragtruth-llama-13b/test"] = lambda scores_df=None: get_ragtruth_dataset(split="test", group='llama-2-13b-chat', filter_length=True)

DATA_LOADER_FN[f"ragtruth-llama-70b/train"] = lambda scores_df=None: get_ragtruth_dataset(split="train", group='llama-2-70b-chat', filter_length=True)
DATA_LOADER_FN[f"ragtruth-llama-70b/trainllm"] = lambda scores_df=None: get_ragtruth_llm_labeled_data(subset="train", group='llama-2-70b-chat')
DATA_LOADER_FN[f"ragtruth-llama-70b/test"] = lambda scores_df=None: get_ragtruth_dataset(split="test", group='llama-2-70b-chat', filter_length=True)

DATA_LOADER_FN[f"ragtruth-all/train"] = lambda scores_df=None: get_ragtruth_dataset(split="train", group=None, filter_length=True)
DATA_LOADER_FN[f"ragtruth-all/trainllm"] = lambda scores_df=None: get_ragtruth_llm_labeled_data(subset="train", group=None)
DATA_LOADER_FN[f"ragtruth-all/trainsync"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None)
DATA_LOADER_FN[f"ragtruth-all/trainsynchconfidence"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="hconfidence")
DATA_LOADER_FN[f"ragtruth-all/trainsynclconfidence"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="lconfidence")
DATA_LOADER_FN[f"ragtruth-all/trainsynchloss"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="hloss")
DATA_LOADER_FN[f"ragtruth-all/trainsynclloss"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="lloss")
DATA_LOADER_FN[f"ragtruth-all/trainsyncnearest"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="nearest")
DATA_LOADER_FN[f"ragtruth-all/trainsyncrandom"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="random")
DATA_LOADER_FN[f"ragtruth-all/trainsynchpplx"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="hpplx")
DATA_LOADER_FN[f"ragtruth-all/trainsynclpplx"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="lpplx")
DATA_LOADER_FN[f"ragtruth-all/trainsynctrainrandom"] = lambda scores_df=None: get_ragtruth_sync_data(subset="train", group=None, filter="trainrandom")
DATA_LOADER_FN[f"ragtruth-all/test"] = lambda scores_df=None: get_ragtruth_dataset(split="test", group=None, filter_length=True)
DATA_LOADER_FN[f"ragtruth-all/testsync"] = lambda scores_df=None: get_ragtruth_sync_data(subset="test", group=None)