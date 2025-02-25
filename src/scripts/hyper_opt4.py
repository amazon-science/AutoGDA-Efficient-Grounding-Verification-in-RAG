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

METRIC="roc"

def perform_run_with_params3(train_dict):
    """ Perform run using the parameters given in train dict. """

    base_config = json.load(open("config_files/hyperparameter_base.json"))

    if args.init_model != "default":
        base_config["initial_generators"][0]["entail_model"] = args.init_model
    if args.num_iters > 0:
        base_config["num_iters"] = args.num_iters
    ## Activate mutations and set corresponding entailment models
    base_config["mutations"][0]["entail_model"] = train_dict["rmiss_model"]
    base_config["mutations"][1]["entail_model"] = train_dict["rmiss_model"]
    base_config["mutations"][2]["entail_model"] = train_dict["rmiss_model"]

    ## Selector
    base_config["selector"]["label_cert_weight"] = train_dict["label_cert_weight"]
    ## Utility
    base_config["utility"]["multiplier"] = train_dict["loss_util_weight"]
    base_config["utility"]["model"] = args.model

    ## Eval model
    base_config["eval_model"] = [args.model]

    if args.dataset == "summedits":
        base_config["initial_generators"][0]["n_per_evidence"] = 16
        base_config["n_select"] = 16
    if args.dataset == "expertqa" or args.dataset == "lfqa-veri":
        base_config["initial_generators"][0]["n_per_evidence"] = 2
        base_config["n_select"] = 2
    runid = f"{hash(frozenset(train_dict.items())):x}"[-12:]
    log_path_base = "hyperopt-f"
    group_name_str = args.group
    log_path = os.path.join(f"{log_path_base}-{args.dataset}-{group_name_str}", runid)
    os.makedirs(log_path, exist_ok=True)
    json.dump(train_dict, open(f"{log_path}/hyperconfig.json", "w"))
    try:
        res = run_from_config(base_config, args.dataset, args.group, runid, save_path=f"{log_path_base}-{args.dataset}-{group_name_str}")
    except Exception as e:
        with open(f"{log_path}/error.txt", "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        res = {METRIC: 0.5}
    return res


def perform_run(trial):
    ## Get values
    hyperparameters = json.load(open("src/scripts/hyperparameters_v3.json"))
    train_dict = {}
    for param in hyperparameters:
        if param["name"] == "rmiss_model" and args.rmiss_model != "default":
            val = args.rmiss_model
        else:
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
            if "bool" in param:
                val = trial.suggest_categorical(param["name"], [True, False])
        train_dict[param["name"]] = val ## Contains values for this run.
    print("Parameters:", train_dict)

    run_result = perform_run_with_params3(train_dict)

    return run_result[METRIC]


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
    parser.add_argument('-g', '--group', type=str, default=None, help="the subgroups of the dataset.")
    parser.add_argument("-m", "--model", type = str, default="tasksource", help="the NLI model to train.")
    parser.add_argument( "--init_model", type=str, default="default", help="the NLI model to assign initial probabilities, otherwise taken from config.")
    parser.add_argument( "--rmiss_model", type=str, default="default", help="use only a specific rmiss model, otherwise treated as a hyperparameter.")
    parser.add_argument("-i", "--num_iters", type=int, default=-1, help="the number of iterations.")
    parser.add_argument("-n", "--n_trials", type=int, default=50, help="how many trials.")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study, storage=f"mysql+pymysql://optuna@localhost/optuna", load_if_exists=True,
        direction="maximize"
    )
    df = study.trials_dataframe()
    print(f"Loaded study {args.study} with {len(df)} trials found.")
    study.optimize(perform_run, n_trials=args.n_trials, timeout=None)
    print("Best:", study.best_params)




