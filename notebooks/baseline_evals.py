import numpy as np
from src.sync_data.population import Population
import pandas as pd

def init_population_from_df(df, labeled=True, use_evidence=None):
    """ Initialize the population with the correct scores. """
    mypopulation = Population()
    if use_evidence is None:
        evidence_items = df["evidence"].unique()
    else:
        evidence_items = use_evidence

    for evidence in evidence_items:
        if labeled:
            for label in [0, 1]:
                index = (df["evidence"] == evidence) & (df["label_binary"] == label)
                use_claims = df[index]["claim"].values
                if "sentence_score" in df.columns:
                    p_init = df[index]["sentence_score"].values
                    if label == 1:
                        p_init[p_init < 0.5] = 0.5
                    else:
                        p_init[p_init > 0.5] = 0.5
                else:
                    p_init = np.ones(len(use_claims))
                mypopulation[(evidence, label)] = use_claims, np.arange(len(use_claims)), np.ones(len(use_claims)), p_init
        else:
            index = (df["evidence"] == evidence)
            mypopulation[evidence] = df[index]["claim"].values
    return mypopulation

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



from src.sync_data.population import Population
def run_evaluators(eval_list: dict, current_w_embed: Population):
    """ Perform evaluation for multiple evaluators. """
    res = {}
    for key, eval_obj in eval_list.items():
        res[key] = eval_obj(current_w_embed, None)
        print(f"Eval result on {key}")
        print(res[key])
    return res


from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.utils.data_utils import get_summedit_group_dataset
import json

for group in ["billsum", "podcast", "news"]:
    print(group)
    dset_train = get_summedit_group_dataset(group=group, subset="train", stratified=True, filter_length=True)
    dset_test = get_summedit_group_dataset(group=group, subset="test", stratified=True, filter_length=True)

    evaluators_same = {}
    evaluators_same_init = {}
    evaluators_other = {}
    evaluators_other_init = {}
    for model_str in ["bart-large", "tasksource", "tasksource_v1"]:
        evaluators_other_init[model_str] = NLIFinetuningEvaluation(dset_test, target_model_str=model_str, num_epochs=0)
        evaluators_same_init[model_str] = NLIFinetuningEvaluation(dset_train, target_model_str=model_str, num_epochs=0)
        evaluators_other[model_str] = NLIFinetuningEvaluation(dset_test, target_model_str=model_str, num_epochs=1)
        evaluators_same[model_str] = NLIFinetuningEvaluation(dset_train, target_model_str=model_str, num_epochs=1)

    pop_train_labeled = init_population_from_df(dset_train.df)

    res_oi = run_evaluators(evaluators_other_init, pop_train_labeled)
    json.dump(res_oi, open(f"runs/c_opt20_{group}/eval_out_init.json", "w"))
    res_si = run_evaluators(evaluators_same_init, pop_train_labeled)
    json.dump(res_si, open(f"runs/c_opt20_{group}/eval_in_init.json", "w"))
    res_oft = run_evaluators(evaluators_other, pop_train_labeled)
    json.dump(res_oft, open(f"runs/c_opt20_{group}/eval_out_finetune.json", "w"))

    for selection_criteria in ["objective", "kld_term"]:
        pop_train_sel_1 = init_population_from_dump(pd.read_csv(f"runs/c_opt20_{group}/data_sel_{selection_criteria}.csv"))
        res_oft = run_evaluators(evaluators_other, pop_train_sel_1)
        json.dump(res_oft, open(f"runs/c_opt20_{group}/eval_out_sel_{selection_criteria}.json", "w"))
        res_sft = run_evaluators(evaluators_same, pop_train_sel_1)
        json.dump(res_sft, open(f"runs/c_opt20_{group}/eval_in_sel_{selection_criteria}.json", "w"))