## Compute NLI Performance of the baselines
import argparse
import json
from src.utils.script_utils import get_query_model_for_dataset, get_datasets, init_population_from_df, get_validation_evidence_size
from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.utils.data_utils import perform_query_integration
import os

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
        target_filename_complete += f"_finetune.json"
    else: # ==0
        target_filename_complete += f"_nofinetune.json"

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

    for current_baseline in baselines:
        if "skip" in current_baseline and current_baseline["skip"] == True:
            continue
        else:
            if "skip" in current_baseline:
                del current_baseline["skip"]
        #if current_baseline["target_model_str"] == "bart-large":
        #    group_train, group_test, group_val = get_datasets(args.dataset, args.group, filter_model_str="facebook/bart-large", length_limit=1024)
        #else:
        group_train, group_test, group_val = get_datasets(args.dataset, args.group)
        target_file, res_dict = get_name_and_current_dict(args, current_baseline)
        current_baseline["query_mode"] = get_query_model_for_dataset(args.dataset, args.group)
        current_baseline["batch_size"] = args.batchsize
        if "lr" not in current_baseline and args.lr > 0.0:
            current_baseline["lr"] = args.lr
        group_test.df = perform_query_integration(group_test.df, mode=current_baseline["query_mode"])
        group_val.df = perform_query_integration(group_val.df, mode=current_baseline["query_mode"])
        val_set_size = get_validation_evidence_size(args.dataset, args.group)
        all_evlist = list(group_val.df.evidence.unique())
        group_val.df = group_val.df[group_val.df.evidence.isin(all_evlist[:val_set_size])]
        print("Using validation set of size", len(group_val))

        popu_train = init_population_from_df(perform_query_integration(group_train.df, mode=current_baseline["query_mode"]), labeled=True)
        if args.split == "val":
            same_ev_eval = NLIFinetuningEvaluation(test_dataset=group_val, **current_baseline)
        else: ## args.split == "test"
            same_ev_eval = NLIFinetuningEvaluation(test_dataset=group_test, **current_baseline)
        if args.group not in res_dict:
            res_dict[args.group] = {}
        res_dict[args.group][current_baseline["target_model_str"]] = same_ev_eval(popu_train, None)
        json.dump(res_dict, open(target_file, "w"))






