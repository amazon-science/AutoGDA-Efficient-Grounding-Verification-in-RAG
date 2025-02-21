import json
import os

from src.utils.script_utils import get_datasets
from src.utils.data_utils import perform_query_integration
from src.sync_data.population import Population
import numpy as np
import pandas as pd
from src.utils.script_utils import init_population_from_df
from src.sync_data.evaluators import PopulationWithEmbeddings
from src.sync_data.combinatorial_optimization import IndependentSelector
from src.sync_data.evaluators import NLIFinetuningEvaluation
import torch
import sys

def run_evaluators(eval_list: dict, current_w_embed: Population):
    """ Perform evaluation for multiple evaluators. """
    res = {}
    for key, eval_obj in eval_list.items():
        res[key] = eval_obj(current_w_embed, None)
        print(f"Eval result on {key}")
        print(res[key])
    return res

def create_topk_selection_dict(popu, res_select, n_select=16):
    selection_dict = {}
    for t in popu.tags:
        selection_dict[t] = np.argsort(-res_select[t])[:n_select]
    return selection_dict
def create_overall_topk_selection_dict(popu, use_idx):
    selection_dict = {}
    idx = 0
    for t in popu.tags:
        sz = len(popu[t])
        if len(use_idx[idx:idx+sz].nonzero().flatten().numpy()) > 0:
            selection_dict[t] = use_idx[idx:idx+sz].nonzero().flatten().numpy()
        idx+=sz
    return selection_dict

def init_population_from_dump(df, labeled=True, use_evidence=None):
    """ Initialize the population with the correct scores. """
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
                p_init = df[index]["p_init"].values
                p_agree = df[index]["p_agree"].values
                if label == 1:
                    p_init[p_init < 0.5] = 0.5
                else:
                    p_init[p_init > 0.5] = 0.5
                mypopulation[(evidence, label)] = use_claims, np.zeros(len(use_claims)), p_agree, p_init
        else:
            index = (df["evidence"] == evidence)
            mypopulation[evidence] = df[index]["claim"].values
    return mypopulation



if __name__ == "__main__":
    lrlog = float(sys.argv[1])
    print(f"Using lr 10^{lrlog}")
    run = "c_opt_indep_ragtqa_nosel2"
    org_data_train, _, org_data_test = get_datasets("ragtruth", group="QA")
    querymode = "queryonly"

    org_data_train.df = perform_query_integration(org_data_train.df, mode=querymode)
    org_data_test.df = perform_query_integration(org_data_test.df, mode=querymode)
    list_evidence = org_data_train.df.evidence.unique()

    synth_data_train = pd.read_csv(f"runs/{run}/data_iteration_0_noselect.csv")
    data_population = init_population_from_dump(synth_data_train, use_evidence=list_evidence)

    pop_target = PopulationWithEmbeddings(init_population_from_df(org_data_train.df, labeled=False, use_evidence=list_evidence))
    data_population = PopulationWithEmbeddings(data_population)

    myselector = IndependentSelector(pop_target, target_radius=0.2, source_radius=0.16, label_cert_weight=20.0)



    eval_models = ["vectara_v2"]
    evaluators_other = {}
    for model_str in eval_models:
        evaluators_other[model_str] = NLIFinetuningEvaluation(org_data_test, target_model_str=model_str,
                                                              num_epochs=1, device="cuda", batch_size=2, lr=np.power(10, lrlog))

    print("computing penalties.")
    penalty_dict = {}
    for t in data_population.tags:
        penalty_dict[t] = myselector._compute_sample_penalties(data_population, t)

    #num_select = [1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28]
    num_select = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28]
    print("Selections:", num_select)

    if os.path.exists(f"runs/{run}/eval_select_strat{lrlog}.json"):
        eval_out_strat = json.load(open(f"runs/{run}/eval_select_strat{lrlog}.json"))
    else:
        eval_out_strat = {}
    if os.path.exists(f"runs/{run}/eval_select_nonstrat{lrlog}.json"):
        eval_out_nonstrat = json.load(open(f"runs/{run}/eval_select_nonstrat{lrlog}.json"))
    else:
        eval_out_nonstrat = {}

    for k in num_select:
        print(f"Selecting {k} ...")
        n_total = int(k * len(penalty_dict))
        ## stratified
        res_selection = {key: -penalty_dict[key].numpy() for key in penalty_dict}
        if k < 1.0:
            val_list, idx_list, labels = [], [], []
            for key in penalty_dict:
                val, idx = torch.min(penalty_dict[key], dim=-1)
                val_list.append(val)
                idx_list.append(idx)
                labels.append(key[1])
            selected_batches0 = torch.argsort(torch.stack([val for val, lab in zip(val_list, labels) if lab == 0]))[
                                :(n_total//2)].numpy()
            selected_batches1 = torch.argsort(torch.stack([val for val, lab in zip(val_list, labels) if lab == 1]))[
                                :(n_total//2)].numpy()
            print(selected_batches0)
            arr_list = []
            l0cnt, l1cnt = 0, 0
            for key in penalty_dict:
                res_all = torch.zeros(len(penalty_dict[key]))
                if key[1] == 0 and l0cnt in selected_batches0:
                    res_all[idx_list[l0cnt]] = 1
                if key[1] == 1 and l1cnt in selected_batches1:
                    res_all[idx_list[l1cnt]] = 1
                if key[1] == 0:
                    l0cnt += 1
                else:
                    l1cnt += 1
                arr_list.append(res_all)
            use_idx = torch.cat(arr_list)
            print(use_idx.shape, torch.sum(use_idx))
            res_selection_dict = create_overall_topk_selection_dict(data_population, use_idx)
            current_w_embed = data_population.get_indexed_subpopulation(res_selection_dict)
            print(current_w_embed.to_dataframe()["tag_1"].mean())
        else:
            res_selection_dict = create_topk_selection_dict(data_population, res_selection, k)
            current_w_embed = data_population.get_indexed_subpopulation(res_selection_dict)

        print("size strat", len(current_w_embed))
        if k not in eval_out_strat:
            eval_out_strat[k] = {}
        eval_out_strat[k].update(run_evaluators(evaluators_other, current_w_embed))
        json.dump(eval_out_strat, open(f"runs/{run}/eval_select_strat{lrlog}.json", "w"))

        ## non-stratified
        all_pens = torch.cat([p for p in penalty_dict.values()])
        print(all_pens.shape)
        use_idx = torch.zeros(len(data_population))
        use_idx[all_pens.argsort(dim=-1)[:n_total]] = 1
        nonstrat_selection_dict = create_overall_topk_selection_dict(data_population, use_idx)
        current_w_embed_ns = data_population.get_indexed_subpopulation(nonstrat_selection_dict)
        print("size nonstrat", len(current_w_embed_ns))
        if k not in eval_out_nonstrat:
            eval_out_nonstrat[k] = {}
        eval_out_nonstrat[k].update(run_evaluators(evaluators_other, current_w_embed_ns))
        json.dump(eval_out_nonstrat, open(f"runs/{run}/eval_select_nonstrat{lrlog}.json", "w"))
