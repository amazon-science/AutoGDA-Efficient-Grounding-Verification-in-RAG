import os
import json
import io
import boto3
import pandas as pd
from src.utils.data_utils import AnnotatedTextDataset
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
## list of files.
LFRQA_FILES ={
    "fiqa": ["fiQA/fiQA_b1_d3_20240205_T1PASS_T2PASS.json", "fiQA/fiQA_b2_d3_20240205_T1PASS_T2PASS.json", "fiQA/fiQA_b3_d1_20231220_T1PASS_T2PASS.json"],
    "lifestyle": ["lifestyle/Lifestyle_Test_D2_20240228_T1PASS_T2PASS.json"],
    "recreation": ["recreation/Recreation_Test_D1_20240129_T1PASS_T2PASS.json"],
    "tech": ["tech/technology_test_d1_20240209_T1PASS_T2PASS.json"]
}

LFRQA_BUCKET = "lfrqa"


def get_json_file_local_or_s3(filepath, localpath="data/lfrqa"):
    """ Load contents from a json file. If the file does not exist locally, content will be loaded from s3.
        :param: filepath the path to the json file relative to localpath.
    """
    path_json = os.path.join(localpath, filepath)
    if not os.path.exists(path_json): ## Download the file from S3
        print(f"Downloading file {filepath} from s3. ")
        s3 = boto3.client("s3")
        obj_json = s3.get_object(Bucket=LFRQA_BUCKET, Key=filepath)
        json_str = io.StringIO(obj_json['Body'].read().decode('utf-8'))
        # create dirs if they are not already there.
        os.makedirs(os.path.dirname(path_json), exist_ok=True)
        obj_local = open(path_json, 'w')
        obj_local.write(json_str.getvalue())
        obj_local.flush()
        obj_local.close()
    return json.load(open(path_json))


def get_num_tokens_cross_encoder(model: CrossEncoder, evidence: str, claim: str):
    return len(model.tokenizer([[evidence, claim]], return_tensors="np")['input_ids'][0])

def filter_by_num_tokens(dataset_df, filter_model_str, length_limit):
    if filter_model_str in ["cross-encoder/nli-deberta-v3-base", "vectara/hallucination_evaluation_model", "allenai/longformer-base-4096"]:
        model = CrossEncoder(filter_model_str, num_labels=1)
        lengths = np.array([get_num_tokens_cross_encoder(model, r[1]["claim"], r[1]["evidence"]) for r in dataset_df.iterrows()])
        dataset_df = dataset_df[lengths <= length_limit]
    else:
        tokenizer = AutoTokenizer.from_pretrained(filter_model_str)
        lengths = np.array([tokenizer.encode(r["evidence"], r["claim"], return_tensors='pt').size(1) for i, r in dataset_df.iterrows()])
        dataset_df = dataset_df[lengths <= length_limit]
    return dataset_df

def get_raw_lfrqa_dataframe(domains=None, share_train=0.8):
    """ Get raw dataframe for lrfqa domains.
        Pass None to get all domains. otherwise pass list of strings or strings. Domains are
        "fiqa", "lifestyle", "recreation", "tech"
    """

    if domains is None:
        domains = list(LFRQA_FILES.keys())
    if isinstance(domains, str):
        domains = [domains]
    # columns to individual rows.
    assign_labels = lambda vals: pd.DataFrame(
        [{"answer": vals["faithful_answer"].iloc[0], "label": 1},
         {"answer": vals["unfaithful_answer"].iloc[0], "label": 0}]).set_index('label')

    df_parts_list = []
    for d in domains:
        for fid, file in enumerate(LFRQA_FILES[d]):
            content = get_json_file_local_or_s3(file, localpath="data/lfrqa")
            ## There are different file formats... check.
            if isinstance(content, dict) and "queries" in content.keys():
                df = pd.DataFrame(content["queries"])
            else:
                df = pd.DataFrame(content)
            df = df[["q", "passages", "faithful_answer", "unfaithful_answer"]]
            # Introduce spliting right here.
            df["is_test"] = 1
            df = df.sample(frac=1, random_state=1)
            n_train = int(len(df) * share_train)
            df.iloc[:n_train, df.columns.get_loc('is_test')] = 0
            # concat passages by semicolon
            df.passages = df.passages.apply(lambda row: "; ".join([t[0] for t in row]))
            # assign labels
            df = df.groupby(["q", "passages", "is_test"]).apply(assign_labels, include_groups=False)
            # undo groupby
            df = df.reset_index()
            ## Add group label
            df["group"] = d
            df["id"] = df.index.to_series().apply(lambda no: f"{d}{fid+1}_{no}")
            df_parts_list.append(df)

    return pd.concat(df_parts_list, axis=0, ignore_index=True)


def get_lfrqa_data(split="train", group=None, filter_length=True,
                   filter_model_str = "tasksource/deberta-base-long-nli",
                   length_limit=1280, share_train=0.8):
    """ :param: split: can be None, train, test.
        :param: group: can be None (return all) or "fiqa", "lifestyle", "recreation", "tech" or list of these
        :param: filter_model_str: the tokenizer to filter out samples exceeding a certain token length.
        Supported models for filtering length:
            "cross-encoder/nli-deberta-v3-base", "vectara/hallucination_evaluation_model", "allenai/longformer-base-4096",
            "tasksource/deberta-base-long-nli", "facebook/bart-large-mnli"
        :param: length_limit: Maximum number of tokens to use (needs to be explicitly specified as some models dont provide bounds).
        :param: share train. share of points used for training. Will be stratified across groups if several groups are used.
    """
    df = get_raw_lfrqa_dataframe(group, share_train=share_train)
    ## Introduce reproducible train_test_split
    if split in ["train", "test"]:
        use_df = df[df.is_test == (0 if split=="train" else 1)]
    else:
        use_df = df

    use_df = use_df.drop("is_test", axis=1)

    ## Rename columns.
    use_df.rename(columns={
        'q': 'query',
        'label': 'label_binary',
        'answer': 'claim',
        'passages': 'evidence'},
        inplace=True)

    use_df["label"] = use_df["label_binary"].apply(lambda label_bin: "ENTAILMENT" if label_bin == 1 else "CONTRADICTION")

    if filter_length:
        use_df = filter_by_num_tokens(use_df, filter_model_str, length_limit)

    if group is not None:
        data_id = "lfrqa" + "-" + str(group)
    else:
        data_id = "lfrqa"
    if split in ["train", "test"]:
        data_id = data_id + "/" + split
    return AnnotatedTextDataset(use_df, data_id)





