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
from src.utils.s3 import load_json_from_s3
from src.utils.constants import _BUCKET_ROOTPATH, _BUCKET_NAME

METRIC="roc"

def perform_run_with_params_mutation(train_dict, mutation_use, seed=1):
    """ Perform run using the parameters given in train dict. """
    base_config = json.load(open("config_files/hyperparameter_base.json"))
    base_config["initial_generators"][0]["generation_seed"] = seed
    ## Activate mutations and set corresponding entailment models
    base_config["mutations"][0]["active"] = (mutation_use == "LLMFillInTheGapsMutation")
    base_config["mutations"][0]["entail_model"] = train_dict["rmiss_model"]
    base_config["mutations"][1]["active"] = (mutation_use == "RephraseMutation")
    base_config["mutations"][1]["entail_model"] = train_dict["rmiss_model"]
    base_config["mutations"][2]["active"] = (mutation_use == "DropSentenceMutation")
    base_config["mutations"][2]["entail_model"] = train_dict["rmiss_model"]
    base_config["skip_init_eval"] = True
    base_config["do_final_eval"] = False

    ## Selector
    base_config["selector"]["label_cert_weight"] = train_dict["label_cert_weight"]
    ## Utility
    base_config["utility"]["multiplier"] = train_dict["loss_util_weight"]

    runid = f"{mutation_use}_seed{seed}"
    res = run_from_config(base_config, args.dataset, args.group, runid, save_path=f"abl_mutation-{args.dataset}-{args.group}")
    print("final result", res)




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
    parser.add_argument("-m", "--model", type = str, default="tasksource", help = "the model to use")
    parser.add_argument("--seed", type=int, default=1, help="random seed.")
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
    for mutation in ["LLMFillInTheGapsMutation", "RephraseMutation", "DropSentenceMutation"]:
        perform_run_with_params_mutation(best_params, mutation, args.seed)



