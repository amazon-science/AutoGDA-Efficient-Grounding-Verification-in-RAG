## Compute NLI Performance of the baselines
import argparse
import json
from src.utils.script_utils import  get_datasets, init_population_from_df
from src.utils.data_utils import get_validation_evidence_size
from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.sync_data.compute_entailments import EntailmentCheckModel
from src.sync_data.population import Population
from tqdm import tqdm
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager

@contextmanager
def suppress():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            with redirect_stderr(null):
                yield

def relabel_population(pop_init, entail_model_str):
    """ Use a different entailment model to relabel the population
        The fine-tuning is subsequently done on the relabeled population.
    """
    mynli = EntailmentCheckModel(entail_model_str)
    ## set p init according to nli model.
    for t in tqdm(pop_init.tags, desc="Recomputing labels"):
        with suppress():
            sent_pairs = [[t[0], sentence] for sentence in pop_init[t]]
            scores = mynli.compute_scores(sent_pairs, show_progress=False)
            pop_init.set_initial_prob(t, scores)

    ## Reassign to tags
    tag_0_list = list([k0 for k0, k1 in pop_init.tags])
    popu_out0 = Population()
    popu_out1 = Population()
    for t in tag_0_list:
        if (t, 0) in pop_init.tags:
            l0 = pop_init.get_initial_prob((t, 0))
            popu_out0[(t, 0)] = pop_init[(t, 0)][l0 <= 0.5]
            popu_out0[(t, 1)] = pop_init[(t, 0)][l0 > 0.5]
        if (t, 1) in pop_init.tags:
            l1 = pop_init.get_initial_prob((t, 1))
            popu_out1[(t, 0)] = pop_init[(t, 1)][l1 <= 0.5]
            popu_out1[(t, 1)] = pop_init[(t, 1)][l1 > 0.5]

    return popu_out0 + popu_out1

def get_name_and_current_dict(args, current_baseline):
    """ Output file name and contents."""
    default_epochs = 1
    default_lr = 1e-5
    if args.split == "val":
        target_filename_complete = f"results/eval_baselines_val_{args.dataset}"
    else:
        target_filename_complete = f"results/eval_baselines_{args.dataset}"
    if "num_epochs" in current_baseline and current_baseline["num_epochs"] != default_epochs and current_baseline["num_epochs"] > 0:
        target_filename_complete += f"_ep{current_baseline['num_epochs']}"
    if "lr" in current_baseline and current_baseline["lr"] != default_lr:
        target_filename_complete += f"_lr{current_baseline['lr']}"
    if "num_epochs" not in current_baseline or current_baseline["num_epochs"] != 0:
        target_filename_complete += f"_finetune"
    else: # ==0
        target_filename_complete += f"_nofinetune"
    if "relabel" in current_baseline:
        target_filename_complete += f"_label_{current_baseline['relabel']}"
    target_filename_complete += ".json"
    if os.path.exists(target_filename_complete):
        ret_dict = json.load(open(target_filename_complete))
    else:
        ret_dict = {}
    return target_filename_complete, ret_dict


if __name__ == "__main__":
    """
    Generates synthetic data from an existing dataset for premises. 
    For each premise it generates a hypothesis and a label from an LLM. 
    """
    os.environ["AWS_REGION_NAME"] = 'us-east-1'
    parser = argparse.ArgumentParser(prog='LLM api', description='', epilog='')
    parser.add_argument('-c', '--baseline_config', required=True, type=str, help='config files with baselines')
    parser.add_argument('-d', '--dataset', type=str, help="the dataset to use")
    parser.add_argument('-g', '--group', type=str, help="the subgroup of the datset.")
    parser.add_argument('-b', '--batchsize', type=int, default=2)
    parser.add_argument("-s", "--split", type=str, default="test", help="which dataset split to use. 'val' or 'test'")
    parser.add_argument("--lr", type=float, default=-1.0, help="The learning rate.")
    args = parser.parse_args()
    baselines = json.load(open(args.baseline_config))
    group_train, group_test, group_val = get_datasets(args.dataset, args.group)
    print(len(group_train), len(group_val), len(group_test))
    val_set_size = get_validation_evidence_size(args.dataset, args.group)
    all_evlist = list(group_val.df.evidence.unique())
    group_val.df = group_val.df[group_val.df.evidence.isin(all_evlist[:val_set_size])]
    print("Using validation set of size", len(group_val))
    popu_train_org = init_population_from_df(group_train.df, labeled=True)
    current_relabel=None
    for current_baseline in baselines:
        if "skip" in current_baseline and current_baseline["skip"] == True:
            continue
        else:
            if "skip" in current_baseline:
                del current_baseline["skip"]
        #if current_baseline["target_model_str"] == "bart-large":
        #    group_train, group_test, group_val = get_datasets(args.dataset, args.group, filter_model_str="facebook/bart-large", length_limit=1024)
        #else:
        target_file, res_dict = get_name_and_current_dict(args, current_baseline)
        current_baseline["batch_size"] = args.batchsize
        if "lr" not in current_baseline and args.lr > 0.0:
            current_baseline["lr"] = args.lr
        #group_test.df = perform_query_integration(group_test.df, mode=current_baseline["query_mode"])
        #group_val.df = perform_query_integration(group_val.df, mode=current_baseline["query_mode"])

        if "relabel" in current_baseline:
            if current_relabel != current_baseline["relabel"]:
                popu_train = relabel_population(popu_train_org, current_baseline["relabel"])
                tag_0_list = list([k0 for k0, k1 in popu_train.tags])
                sum0 = sum([len(popu_train[(t, 0)]) for t in tag_0_list if (t, 0) in popu_train.tags])
                sum1 = sum([len(popu_train[(t, 1)]) for t in tag_0_list if (t, 1) in popu_train.tags])
                print("Labeled 0:", sum0, "Labeled 1:", sum1)
                current_relabel = current_baseline["relabel"]
            else:
                # Already relabeled
                print("Reusing relabeled population from previous run. ")
            del current_baseline["relabel"]
        else:
            popu_train = popu_train_org
        try:
            if args.split == "val":
                same_ev_eval = NLIFinetuningEvaluation(test_dataset=group_val, **current_baseline)
            else: ## args.split == "test"
                same_ev_eval = NLIFinetuningEvaluation(test_dataset=group_test, **current_baseline)
            if args.group not in res_dict:
                res_dict[args.group] = {}
            res_dict[args.group][current_baseline["target_model_str"]] = same_ev_eval(popu_train, None)
            json.dump(res_dict, open(target_file, "w"))
        except Exception as e:
            print(e)
            raise e






