import json
import os

from src.scripts.script_utils import get_datasets
from src.utils.data_utils import perform_query_integration
from src.sync_data.population import Population
import numpy as np
import pandas as pd
from src.scripts.script_utils import init_population_from_df
from src.sync_data.evaluators import PopulationWithEmbeddings
from src.sync_data.combinatorial_optimization import SGDOptimizationSelector, IndependentSelector
from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.sync_data.utility_functions import ModelLossUtility
import torch
import sys
import optuna

## parameters
n_select = 2

## global vars
hyperparameters = None
pop_target = None
data_population = None
pop_train = None
model_arch = None

def create_topk_selection_dict(popu, res_select, n_select=16):
    selection_dict = {}
    for t in popu.tags:
        selection_dict[t] = np.argsort(-res_select[t])[:n_select]
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


def perform_run(trial):
    run = "c_opt_indep_ragtqa_nosel2"
    ## Get values
    train_dict = {}
    for param in hyperparameters:
        val = None
        if "options" in param:
            val = trial.suggest_categorical(param["name"],
                                            list([tuple(p) if isinstance(p, list) else p for p in param["options"]]))
        if "logrange" in param:
            val = trial.suggest_float(param["name"], param["logrange"][0], param["logrange"][1], log=True)
        if "floatrange" in param:
            val = trial.suggest_float(param["name"], param["floatrange"][0], param["floatrange"][1])
        if "intrange" in param:
            val = trial.suggest_int(param["name"], param["intrange"][0], param["intrange"][1])
        train_dict[param["name"]] = val ## Contains values for this run.
    train_dict["target_radius"] = train_dict["target_radius_ratio"]*train_dict["source_radius"]
    print("Parameters:", train_dict)
    myeval = NLIFinetuningEvaluation(org_data_test, target_model_str=model_arch,
                                     num_epochs=1, device="cuda", batch_size=2, lr=1e-5)
    model, tok = myeval._prepare_model()
    my_util = ModelLossUtility(model, tok, multiplier=train_dict["loss_util_weight"])
    myselector = IndependentSelector(pop_target, target_radius=train_dict["target_radius"],
                                     source_radius=train_dict["source_radius"],
                                     label_cert_weight=train_dict["label_cert_weight"],
                                     utility_fn=my_util)

    ## Execute selection
    res_selection = myselector.select(data_population, n_select)
    res_selection_dict = create_topk_selection_dict(data_population, res_selection, n_select)
    current_w_embed = data_population.get_indexed_subpopulation(res_selection_dict)
    print("Size of selected population: ", len(current_w_embed))
    ## Finetune and eval
    run_result = myeval(current_w_embed, None)

    print("Obtained eval result", run_result)
    ## Append to logfile
    if os.path.exists(f"runs/{run}/hyperopt_{model_arch}.json"):
        runs_list = json.load(open(f"runs/{run}/hyperopt_{model_arch}.json"))
    else:
        runs_list = []

    run_result["parameters"] = train_dict
    runs_list.append(run_result)
    json.dump(runs_list, open(f"runs/{run}/hyperopt_{model_arch}.json", "w"))
    return run_result["roc"]

if __name__ == "__main__":
    model_arch = sys.argv[1]
    print(f"Using model {model_arch}")
    run = "c_opt_indep_ragtqa_nosel2"
    hyperparameters = json.load(open("src/scripts/hyperparameter_ranges.json"))
    org_data_train, _, org_data_test = get_datasets("ragtruth", group="QA")
    querymode = "queryonly"
    org_data_train.df = perform_query_integration(org_data_train.df, mode=querymode)
    org_data_test.df = perform_query_integration(org_data_test.df, mode=querymode)

    synth_data_train = pd.read_csv(f"runs/{run}/data_iteration_0_noselect.csv")
    list_evidence = list(org_data_train.df.evidence.unique()) #[:10]
    data_population = init_population_from_dump(synth_data_train, use_evidence=list_evidence)

    pop_target = PopulationWithEmbeddings(init_population_from_df(org_data_train.df, labeled=False,
                                                                  use_evidence=list_evidence))
    data_population = PopulationWithEmbeddings(data_population)

    study = optuna.create_study(direction="maximize")
    study.optimize(perform_run, n_trials=50, timeout=None)
