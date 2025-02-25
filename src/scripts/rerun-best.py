# A script performing finetuning on target data of different sized to find a suitable validation set size that
# is still insufficient for finetuning (otherwise it would make no sense to have the unsupervised approach)
import argparse
import json
import traceback

## setup a test population
import torch
from src.sync_data.generation_loop import run_from_config
import optuna
import os
from src.scripts.reevaluate_test import reevaluate
from src.utils.script_utils import get_datasets
from src.utils.s3 import load_json_from_s3
from src.utils.constants import _BUCKET_NAME, _BUCKET_ROOTPATH
def get_log_path_params(train_dict):
    if args.rmiss is not None: # Take model from hyper opt otherwise.
        train_dict["rmiss_model"] = args.rmiss
    log_path_base = "eval_run" + args.run_folder_suffix
    log_path_base = f"{log_path_base}-{args.study}"
    #log_path_base = f"{log_path_base}-{args.dataset}-{args.group}"
    train_dict["target_model"] = args.target_model
    run_id = f"seed_{args.seed}-{train_dict['rmiss_model']}-{train_dict['target_model']}"
    train_dict["num_iters"] = args.num_iters

    if args.pinit_model is not None:
        train_dict["pinit_model"] = args.pinit_model
    else:
        train_dict["pinit_model"] = "vectara_v2"

    return train_dict, log_path_base, run_id, train_dict["num_iters"]

def perform_run_with_params3(train_dict, log_path_base, run_id):
    """ Perform run using the parameters given in train dict. """
    log_path = os.path.join(log_path_base, run_id)
    base_config = json.load(open(args.config_file))
    base_config["num_iters"] = train_dict["num_iters"]
    base_config["initial_generators"][0]["entail_model"] = train_dict["pinit_model"]
    base_config["initial_generators"][0]["generation_seed"] = args.seed
    base_config["skip_init_eval"] = True
    ## Utility
    if "utility" in base_config:
        base_config["utility"]["model"] = args.target_model
        base_config["utility"]["multiplier"] = train_dict["loss_util_weight"]
    base_config["eval_model"] = [args.target_model]
    ## Activate mutations and set corresponding entailment models
    base_config["mutations"][0]["entail_model"] = train_dict["rmiss_model"]
    base_config["mutations"][1]["entail_model"] = train_dict["rmiss_model"]
    base_config["mutations"][2]["entail_model"] = train_dict["rmiss_model"]

    ## Selector
    if base_config["selector"]["class"] != "RandomSelector":
        base_config["selector"]["label_cert_weight"] = train_dict["label_cert_weight"]




    if args.dataset == "summedits":
        base_config["initial_generators"][0]["n_per_evidence"] = 16
        base_config["n_select"] = 16
        base_config["n_evidence"] = 16
    if args.dataset == "expertqa" or args.dataset == "lfqa-veri":
        base_config["initial_generators"][0]["n_per_evidence"] = 2
        base_config["n_select"] = 2

    os.makedirs(log_path, exist_ok=True)
    json.dump(train_dict, open(f"{log_path}/hyperconfig.json", "w"))
    run_from_config(base_config, args.dataset, args.group, run_id, save_path=log_path_base)
    return log_path


if __name__ == "__main__":
    """
    Generates synthetic data from an existing dataset for premises. 
    For each premise it generates a hypothesis and a label from an LLM. 
    """
    os.environ["AWS_REGION_NAME"] = 'us-east-1'
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(prog='LLM api', description='', epilog='')
    parser.add_argument('-s', '--study', type=str, help="name of the study")
    parser.add_argument('-d', '--dataset', type=str, default="ragtruth",  help="the dataset to use")
    parser.add_argument('-g', '--group', type=str, default="QA", help="the subgroup of the dataset.")
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--rmiss", type=str, default=None, help="model to use to compute pmiss. Default will be taken from best hyperparameters.")
    parser.add_argument("--pinit_model", type=str, default=None, help="model to use for computing initial prob. Default will be vectara_v2.")
    parser.add_argument("-t", "--target_model", type=str, default="tasksource", help="model to finetune and compute utility with")
    parser.add_argument("-i", "--num_iters", type=int, default=1, help="number of iterations. Default 1.")
    parser.add_argument('--config_file', type=str, default="config_files/hyperparameter_base.json", help="path to config file")
    parser.add_argument('--run_folder_suffix', type=str, default="", help="naming suffix for the folder where the runs are saved")
    parser.add_argument('--no_reeval', help="evaluate the final data generated with other models after finishing run.", default=False, action='store_true')
    args = parser.parse_args()
    best_params=None
    try:
        study = optuna.load_study(study_name=args.study, storage=f"mysql+pymysql://optuna@localhost/optuna")
        best_params = study.best_params
    except Exception as e:
        print("Couldn't load study locallay, retrying S3. Error: ", e)

    if best_params is None:
        all_hyperparams = load_json_from_s3(_BUCKET_NAME, _BUCKET_ROOTPATH + "/results/hyperparameters.json")
        if args.study in all_hyperparams:
            best_params = all_hyperparams[args.study]
        else:
            raise ValueError(f"Failed to load study {args.study} locally or from S3. Please use an existing study.")

    print("Best:", best_params)

    train_dict, log_path_base, run_id, n_iter = get_log_path_params(dict(**best_params))
    print("Using parameters:", train_dict)
    log_path = os.path.join(log_path_base, run_id)
    ## If run has been already completed only perform final eval.
    skip = False
    if os.path.exists(f"{log_path}/eval_out_list.json"):
        res_dict = json.load(open((f"{log_path}/eval_out_list.json"), "r"))
        if str(n_iter) in res_dict:
            skip = True
    if not skip:
        print("Performing run ...")
        perform_run_with_params3(train_dict, log_path_base, run_id)
    else:
        print("Skipping run ...")

    ## Compute other models.
    if not args.no_reeval:
        org_data_train, org_data_test, org_data_val = get_datasets(args.dataset, group=args.group)
        reevaluate(log_path, ["tasksource", "flan-t5-base", "bart-large"], org_data_test, useiters=[n_iter])
    #reevaluate(log_path, ["bart-large"], org_data_test, useiters=[n_iter])


