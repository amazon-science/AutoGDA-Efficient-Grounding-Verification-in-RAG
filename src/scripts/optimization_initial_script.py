import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
#torch.multiprocessing.set_start_method('spawn')

import pandas as pd
## Load initial data
from src.utils.data_utils import get_ragtruth_dataset, get_summedit_group_dataset
from src.sync_data.pc_mutations import LLMFillInTheGapsMutation
from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.sync_data.combinatorial_optimization import SGDOptimizationSelector, IndependentSelector
from script_utils import get_datasets, init_population_from_df, get_query_model_for_dataset
## Initialize a population and compute embeddings
from src.sync_data.population import Population
from src.utils.data_utils import perform_query_integration
import numpy as np
import json
import argparse
from src.sync_data.evaluators import PopulationWithEmbeddings

def parse_args():
    parser = argparse.ArgumentParser(prog='Synthetic Data Generation', description='', epilog='')
    parser.add_argument('-d', '--dataset', type=str, default='ragtruth')
    parser.add_argument('-e', '--evidences', type=int, default=10, help="Number of evidences to use, -1 means use all.")
    parser.add_argument("--n_select", type=int, default=16, help="how many samples to use per evidence tag.")
    parser.add_argument('--ft_batchsize', type=int, default=2, help='Batch size used for finetuning.')
    parser.add_argument('-m', '--eval_model', choices=['tasksource_v1', 'tasksource', 'vectara_v1', 'bart-large', "vectara_v2"], default='tasksource', nargs="+")
    parser.add_argument('--num_iters', type=int, default=3)
    parser.add_argument('-r', '--region_name', choices=['us-east-1', 'us-east-2', 'us-west-1'], default='us-east-1')
    parser.add_argument('--run', type=str, default="c_opt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_init_eval", type=bool, default=False)
    args = parser.parse_args()
    return args

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

def run_evaluators(eval_list: dict, current_w_embed: Population):
    """ Perform evaluation for multiple evaluators. """
    res = {}
    for key, eval_obj in eval_list.items():
        res[key] = eval_obj(current_w_embed, None)
        print(f"Eval result on {key}")
        print(res[key])
    return res

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    org_data_train, org_data_test, synth_data_train = None, None, None
    querymode = None
    if args.dataset == "ragtruth":
        # Load initial data
        org_data_train, org_data_test = get_datasets("ragtruth", group="Summary")
        synth_data_train = pd.read_csv("sync_data/d16_ragtruth_FULL.csv")
    elif args.dataset == "ragtruth-qa":
        org_data_train, org_data_test = get_datasets("ragtruth", group="QA")
        querymode = "queryonly"
        synth_data_train = pd.read_csv("sync_data/d16_ragtruthqa_FULL.csv")
    elif "summedits" in args.dataset:
        parts = args.dataset.split("-")
        org_data_train = get_summedit_group_dataset(group=parts[1], subset="train", stratified=True, filter_length=True)
        org_data_test = get_summedit_group_dataset(group=parts[1], subset="test", stratified=True, filter_length=True)
        synth_data_train = pd.read_csv(f"sync_data/d16_{args.dataset}.csv")
    elif "expertqa" in args.dataset:
        parts = args.dataset.split("-")
        org_data_train, org_data_test = get_datasets("expertqa", group=parts[1])
        querymode = "prepend"
        synth_data_train = pd.read_csv(f"sync_data/d4_{args.dataset}.csv")
    else:
        raise ValueError("Other datasets currently not supported.")

    ## Select evidences to use: This will determine the size of the subpopulations.
    list_evidence = synth_data_train.evidence.unique()[:args.evidences]
    print("Used evidences:", len(list_evidence))
    org_data_train.df = perform_query_integration(org_data_train.df, mode=querymode)
    org_data_train.df = org_data_train.df[org_data_train.df['evidence'].isin(list_evidence)]
    org_data_test.df = perform_query_integration(org_data_test.df, mode=querymode)
    #rg_data_test.df = org_data_test.df.iloc[:100]


    ## Setup mutations: ToDo config file
    llmmutations = LLMFillInTheGapsMutation(device=args.device, n_mutations=3, model_name="claude3-haiku", batch_size=10,
                                            temperature=1.0, mask_output_perc=0.2, connected=True, n_multiply=2, n_threads=4,
                                            preserve_meaning=True, use_entailment=True, entail_model="tasksource")



    ## Setup evaluators
    evaluators_same = {}
    evaluators_other = {}
    for model_str in args.eval_model:
        evaluators_same[model_str] = NLIFinetuningEvaluation(org_data_train, target_model_str=model_str,
                                                             num_epochs=1, device=args.device, batch_size=args.ft_batchsize)
        evaluators_other[model_str] = NLIFinetuningEvaluation(org_data_test, target_model_str=model_str,
                                                             num_epochs=1, device=args.device, batch_size=args.ft_batchsize)

    ## Initialize Populations
    pop_target = init_population_from_df(org_data_train.df, labeled=False, use_evidence=list_evidence)
    pop_init = init_population_from_df(synth_data_train, use_evidence=list_evidence, max_per_evidence=args.n_select)
    print("Population sizes: ", len(pop_target), len(pop_init))

    # Computing Embeddings
    print("Computing intial embeddings.")
    target_w_emb = PopulationWithEmbeddings(pop_target)
    current_w_embed = PopulationWithEmbeddings(pop_init)

    #myselector = SGDOptimizationSelector(target_w_emb, target_radius=0.2, source_radius=0.16,
    #                                     sgd_steps=100, sgd_lr=5e-2, label_cert_weight=10.0)

    myselector = IndependentSelector(target_w_emb, target_radius=0.2, source_radius=0.16, label_cert_weight=20.0)

    metric_list = []
    objectives_list = []
    eval_out_list = []
    eval_in_list = []
    log_path = os.path.join("runs", args.run)
    os.makedirs(log_path, exist_ok=True)

    for i in range(args.num_iters):
        if i > 0 or args.skip_init_eval == False:
            print(f"Evaluating on other evidences.")
            out_eval = run_evaluators(evaluators_other, current_w_embed)
            print(out_eval)
            eval_out_list.append(out_eval)
            print(f"Evaluating on same evidences.")
            #in_eval = run_evaluators(evaluators_same, current_w_embed)
            #print(in_eval)
            #eval_in_list.append(in_eval)
            print(f"ITERATION {i}")
            print("Population size:", current_w_embed.get_total_size())

            ## Initial Eval
            res_metrics = evaluate_dist_certainties(current_w_embed, target_w_emb)
            metric_list.append(res_metrics)
            for k, v in res_metrics.items():
                print(f"{k}: {np.array(v).mean()}")
            res_objective = myselector.evaluate_objective(current_w_embed)
            print(list(res_objective.values()))
            objectives_list.append(list(res_objective.values()))

            json.dump(eval_in_list, open(f"{log_path}/eval_in_list.json", "w"))
            json.dump(eval_out_list, open(f"{log_path}/eval_out_list.json", "w"))
            json.dump(objectives_list, open(f"{log_path}/eval_objectives.json", "w"))
            json.dump(metric_list, open(f"{log_path}/eval_metrics.json", "w"))
            current_w_embed.to_dataframe().to_csv(f"{log_path}/data_iteration_{i}.csv", index=False)

        else:
            if os.path.exists(f"{log_path}/eval_in_list.json"):
                eval_in_list = json.load(open(f"{log_path}/eval_in_list.json"))
            if os.path.exists(f"{log_path}/eval_out_list.json"):
                eval_out_list = json.load(open(f"{log_path}/eval_out_list.json"))
            if os.path.exists(f"{log_path}/eval_objectives.json"):
                objectives_list = json.load(open(f"{log_path}/eval_objectives.json"))
            if os.path.exists(f"{log_path}/eval_metrics.json"):
                metric_list = json.load(open(f"{log_path}/eval_metrics.json"))

        ## Mutation
        pop_update_new = PopulationWithEmbeddings(llmmutations.mutate_all_tags(current_w_embed))

        res_metrics = evaluate_dist_certainties(pop_update_new, target_w_emb)
        metric_list.append(res_metrics)
        for k, v in res_metrics.items():
            print(f"{k}: {np.array(v).mean()}")

        current_w_embed = current_w_embed + pop_update_new
        current_w_embed.to_dataframe().to_csv(f"{log_path}/data_iteration_{i}_noselect.csv", index=False)
        exit(0)
        ## Selection
        res_selection = myselector.select(current_w_embed, 16)
        res_selection_dict = create_topk_selection_dict(current_w_embed, res_selection, args.n_select)
        current_w_embed = current_w_embed.get_indexed_subpopulation(res_selection_dict)

    ## FINAL EVAL
    print(f"Evaluating on other evidences.")
    out_eval = run_evaluators(evaluators_other, current_w_embed)
    print(out_eval)
    eval_out_list.append(out_eval)
    print(f"Evaluating on same evidences.")
    #in_eval = run_evaluators(evaluators_same, current_w_embed)
    #print(in_eval)
    #eval_in_list.append(in_eval)
    print(f"ITERATION {i}")
    print("Population size:", current_w_embed.get_total_size())
    ## Initial Eval
    res_metrics = evaluate_dist_certainties(current_w_embed, target_w_emb)
    metric_list.append(res_metrics)

    res_objective = myselector.evaluate_objective(current_w_embed)
    print(list(res_objective.values()))
    objectives_list.append(list(res_objective.values()))

    json.dump(eval_in_list, open(f"{log_path}/eval_in_list.json", "w"))
    json.dump(eval_out_list, open(f"{log_path}/eval_out_list.json", "w"))
    json.dump(metric_list, open(f"{log_path}/eval_metrics.json", "w"))
    json.dump(objectives_list, open(f"{log_path}/eval_objectives.json", "w"))
    current_w_embed.to_dataframe().to_csv(f"{log_path}/data_iteration_{args.num_iters}.csv", index=False)