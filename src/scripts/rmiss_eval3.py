from src.sync_data.generation_loop import run_from_config
import argparse
import torch
import json
import optuna
import traceback
from src.utils.s3 import load_json_from_s3
from src.utils.constants import _BUCKET_NAME, _BUCKET_ROOTPATH

def parse_args():
    parser = argparse.ArgumentParser(prog='Synthetic Data Generation', description='', epilog='')
    parser.add_argument('-d', '--dataset', type=str, default='ragtruth')
    parser.add_argument('-g', '--group', type=str, default='Summary', help='subgroup of the dataset')
    parser.add_argument('--evidences', type=int, default=-1, help="Number of evidences to use, -1 means use all.")
    parser.add_argument("--n_select", type=int, default=4, help="how many samples to select per evidence/label tag.")
    parser.add_argument("--rmiss_model", type=str, default="tasksource", help="which model to use for rmiss")
    parser.add_argument("--seed", type=int, default=2, help="which run seed to use")
    parser.add_argument("--study", type=str, default='rtsumm_tasksource_final2', help="which study to use")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    config = json.load(open("config_files/hyperparameter_base.json"))
    config["evidences"] = args.evidences
    config["n_select"] = args.n_select
    best_params = None
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

    run_name = f"rmiss_eval3_v{args.seed}_{args.rmiss_model}_{args.dataset}_{args.group}_rerun"
    ## Replace all entail models in config.
    for m in config["mutations"]:
        if args.rmiss_model != "none":
            m["entail_model"] = args.rmiss_model
        else:
            m["entail_model"] = None
            config["single_rmiss"] = False
    # use best hyperparams
    config["selector"]["label_cert_weight"] = best_params["label_cert_weight"]
    config["utility"]["multiplier"] = best_params["loss_util_weight"]
    config["initial_generators"][0]["generation_seed"] = args.seed
    config["do_final_eval"] = True
    try:
        run_from_config(config, args.dataset, args.group, run_name, save_path="rmiss-ragtruth-Summary")
    except Exception as e:
        with open(f"rmiss-ragtruth-Summary/{run_name}/error.txt", "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())