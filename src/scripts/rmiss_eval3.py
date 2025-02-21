from src.sync_data.generation_loop import run_from_config
import argparse
import torch
import json

def parse_args():
    parser = argparse.ArgumentParser(prog='Synthetic Data Generation', description='', epilog='')
    parser.add_argument('-d', '--dataset', type=str, default='ragtruth')
    parser.add_argument('-g', '--group', type=str, default='QA', help='subgroup of the dataset')
    parser.add_argument('--evidences', type=int, default=-1, help="Number of evidences to use, -1 means use all.")
    parser.add_argument("--n_select", type=int, default=4, help="how many samples to select per evidence/label tag.")
    parser.add_argument("--rmiss_model", type=str, default="tasksource", help="which model to use for rmiss")
    parser.add_argument("--seed_version", type=int, default=2, help="which run seed to use")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    seed_ids = {0: 3, 1: 2, 2: 1, 3: 42, 4: 43}
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    config = json.load(open("config_files/rmiss_config.json"))
    config["evidences"] = args.evidences
    config["n_select"] = args.n_select
    run_name = f"rmiss_eval3_v{args.seed_version}_{args.rmiss_model}_{args.dataset}_{args.group}_rerun"
    ## Replace all entail models in config.
    for m in config["mutations"]:
        if args.rmiss_model != "none":
            m["entail_model"] = args.rmiss_model
        else:
            m["entail_model"] = None
    config["initial_generators"][0]["generation_seed"] = seed_ids[args.seed_version]

    run_from_config(config, args.dataset, args.group, run_name)
