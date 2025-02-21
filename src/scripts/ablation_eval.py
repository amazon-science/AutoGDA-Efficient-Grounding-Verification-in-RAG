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
    parser.add_argument("--ft_model", type=str, default="tasksource", help="which model to use for finetuning.")
    parser.add_argument("--rmiss_model", type=str, default="tasksource", help="which model to use for rmiss")
    parser.add_argument("--seed_version", type=int, default=2, help="which run seed to use")
    parser.add_argument("--skip_init_eval", type=bool, default=False, help="skip initial evaluation")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    seed_ids = {0: 3, 1: 2, 2: 1, 3: 42, 4: 43}
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    config = json.load(open("config_files/ablation_config.json"))
    config["evidences"] = args.evidences
    config["n_select"] = args.n_select
    config["skip_init_eval"] = args.skip_init_eval
    #run_name = f"ablation_seed{args.seed_version}_{args.rmiss_model}_{args.ft_model}_{args.dataset}_{args.group}"
    run_name = f"ablation_s2_{args.rmiss_model}_{args.ft_model}_{args.dataset}_{args.group}"
    select_max_cert = False
    ## Replace all entail models in config.
    for m in config["mutations"]:
        if args.rmiss_model != "none":
            m["entail_model"] = args.rmiss_model
        else:
            m["entail_model"] = None
    config["utility"]["model"] = args.ft_model
    config["eval_model"] = [args.ft_model]
    config["initial_generators"][0]["generation_seed"] = seed_ids[args.seed_version]

    run_from_config(config, args.dataset, args.group, run_name)
