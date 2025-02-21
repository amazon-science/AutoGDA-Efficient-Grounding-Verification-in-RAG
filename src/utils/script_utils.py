## Initialize a population and compute embeddings
import selectors

from src.sync_data.population import Population
import numpy as np
from copy import deepcopy
import torch
from src.utils.data_utils import get_ragtruth_dataset, get_expertqa_data, get_summedit_group_dataset

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_datasets(base_dataset = "ragtruth", group="Summary", filter_model_str="tasksource/deberta-base-long-nli", length_limit=1280):
    """ Return train and test partition of datasets. """
    data_val = None
    if base_dataset == "ragtruth":
        if group=="Summary" or group =="QA":
            data_test = get_ragtruth_dataset(split="test", group=None, filter_length=True, task=group,
                                              filter_model_str=filter_model_str, length_limit=length_limit)
            data_train = get_ragtruth_dataset(split="train", group=None, filter_length=True, task=group,
                                              filter_model_str=filter_model_str, length_limit=length_limit)
            data_val = get_ragtruth_dataset(split="val", group=None, filter_length=True, task=group,
                                              filter_model_str=filter_model_str, length_limit=length_limit)
        else:
            raise ValueError("RAGTruth-Group must be either 'Summary' or 'QA'")
    elif base_dataset == "summedits":
        data_test = get_summedit_group_dataset(subset="test", group=group, filter_length=True,
                                          filter_model_str=filter_model_str, length_limit=length_limit)
        data_train = get_summedit_group_dataset(subset="train", group=group, filter_length=True,
                                          filter_model_str=filter_model_str, length_limit=length_limit)
    elif base_dataset == "expertqa":
        data_test = get_expertqa_data(subset="test", group=group, filter_length=True,
                                       filter_model_str=filter_model_str, length_limit=length_limit)
        data_train = get_expertqa_data(subset="train", group=group, filter_length=True,
                                        filter_model_str=filter_model_str, length_limit=length_limit)
    else:
        raise ValueError(f"Unknown dataset {base_dataset}")
    return data_train, data_test, data_val


def get_validation_evidence_size(base_dataset, group):
    """ Get the number of evidences to use in a reasonable validation dataset (eg. ca. 100 labeled examples in total). """
    dict_vals = {("ragtruth", "QA"): 24, ("ragtruth", "Summary"): 24}
    if (base_dataset, group) in dict_vals:
        return dict_vals[(base_dataset, group)]
    else:
        return None

def get_query_model_for_dataset(base_dataset = "ragtruth", group="Summary"):
    if base_dataset == "ragtruth":
        if group=="Summary":
            return None
        elif group=="QA":
            return "queryonly"
    if base_dataset == "summedits":
        return None
    if base_dataset == "expertqa":
        return "prepend"
    raise ValueError(f"Unknown dataset or group {base_dataset}/{group}")

def init_population_from_df(df, labeled=True, use_evidence=None, max_per_evidence=None, seed=1):
    """ Initialize the population with the correct scores. """
    mypopulation = Population()
    if use_evidence is None:
        evidence_items = df["evidence"].unique()
    else:
        evidence_items = use_evidence
    print(len(evidence_items))
    for evidence in evidence_items:
        if labeled:
            for label in [0, 1]:
                index = (df["evidence"] == evidence) & (df["label_binary"] == label)
                use_claims = df[index]["claim"].values
                if len(use_claims) > 0 :
                    if "sentence_score" in df.columns or "p_init" in df.columns:
                        if "sentence_score" in df.columns:
                            #print("Getting initial scores from sentence_score.")
                            p_init = df[index]["sentence_score"].values
                        else: # p_init
                            #print("Getting initial scores from p_init.")
                            p_init = df[index]["p_init"].values
                        if label == 1:
                            p_init[p_init < 0.5] = 0.5
                        else:
                            p_init[p_init > 0.5] = 0.5
                    else:
                        p_init = np.ones(len(use_claims))
                    mypopulation[(evidence, label)] = use_claims, np.arange(len(use_claims)), np.ones(len(use_claims)), p_init
        else:
            index = (df["evidence"] == evidence)
            sub_df = df[index]["claim"].values
            if len(sub_df) > 0:
                mypopulation[evidence] = sub_df
        if max_per_evidence is not None:
            mypopulation = mypopulation.sample(max_per_evidence, seed=seed)
    return mypopulation


def init_population_from_dump(df, labeled=True, use_evidence=None, max_per_evidence=None, seed=1):
    """ Initialize the population with the correct scores from a dump df by the algorithm
        (this is similar to init_population_from_df, but has different column naming).
    """
    mypopulation = Population()
    if use_evidence is None:
        evidence_items = df["tag_0"].unique()
    else:
        evidence_items = use_evidence

    for evidence in evidence_items:
        if labeled:
            for label in [0, 1]:
                index = (df["tag_0"] == evidence) & (df["tag_1"] == label)
                use_claims = df[index]["sample"].values
                if len(use_claims) > 0:
                    p_init = df[index]["p_init"].values
                    if label == 1:
                        p_init[p_init < 0.5] = 0.5
                    else:
                        p_init[p_init > 0.5] = 0.5

                    if "p_agree" in df.columns:
                        p_agree = df[index]["p_agree"].values
                    else:
                        p_agree = np.ones(len(use_claims))

                    mypopulation[(evidence, label)] = use_claims, np.arange(len(use_claims)), p_agree, p_init
        else:
            index = (df["evidence"] == evidence)
            mypopulation[evidence] = df[index]["claim"].values
        if max_per_evidence is not None:
            mypopulation = mypopulation.sample(max_per_evidence, seed=seed)
    return mypopulation


def convert_to_json_dict(res_select):
    dict_save = {}
    for key in res_select:
        dict_save[key[0]+"_"+str(key[1])] = list(res_select[key].astype(float))
    return dict_save

def create_random_selection_dict(popu, n_select=16):
    selection_dict = {}
    for t in popu.tags:
        selection_dict[t] = np.random.permutation(len(popu[t]))[:n_select]
    return selection_dict

def create_topk_selection_dict(popu, res_select, n_select=16):
    selection_dict = {}
    for t in popu.tags:
        selection_dict[t] = np.argsort(-res_select[t])[:n_select]
    return selection_dict

from src.sync_data.combinatorial_optimization import get_label_correctness_kld


def evaluate_dist_certainties(population_w_emb, targets_w_emb):
    kld_terms_list = []
    agree_prob_list = []
    min_avg_dist_list = []
    for t in population_w_emb.tags:
        kld_term = get_label_correctness_kld(population_w_emb.get_agreement(t), population_w_emb.get_initial_prob(t))
        kld_terms_list.append(kld_term.mean())
        agree_prob_list.append(population_w_emb.get_agreement(t).mean())

        ## Distance to target embedding
        org_dists = torch.cdist(targets_w_emb.get_embeddings(t[0]), population_w_emb.get_embeddings(t))
        # print(org_dists.shape)
        min_dists = torch.min(org_dists, dim=0)[0]
        min_avg_dist_list.append(min_dists.mean().item())
    return {"kld_term": kld_terms_list, "p_agree": agree_prob_list,
            "min_dist": min_avg_dist_list}


def run_evaluators(eval_list: dict, current_w_embed: Population):
    """ Perform evaluation for multiple evaluators. """
    res = {}
    for key, eval_obj in eval_list.items():
        res[key] = eval_obj(current_w_embed, None)
        print(f"Eval result on {key}")
        print(res[key])
    return res