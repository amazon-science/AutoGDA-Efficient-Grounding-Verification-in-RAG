import os
import argparse
from src.sync_data.compute_entailments import EntailmentCheckModel
from src.utils.script_utils import get_datasets
import json
import time
import pandas as pd

if __name__ == "__main__":
    """
    Generates synthetic data from an existing dataset for premises. 
    For each premise it generates a hypothesis and a label from an LLM. 
    """
    os.environ["AWS_REGION_NAME"] = 'us-east-1'
    parser = argparse.ArgumentParser(prog='LLM api', description='', epilog='')
    parser.add_argument('-d', '--dataset', type=str, default="ragtruth",  help="the dataset to use")
    parser.add_argument('-g', '--group', type=str, default="Summary", help="the subgroups of the dataset.")
    parser.add_argument("-m", "--model", type =str, default=None, nargs="+", help="the NLI model to test.")
    parser.add_argument( "--n_samples", type=int, default=100, help="number of samples to use")
    parser.add_argument("-r", "--n_retries", type=int , default=5, help="number of retries")
    args = parser.parse_args()

    log_file = f"results/timing_log_{args.n_samples}_{args.dataset}_{args.group}.json"
    if os.path.exists(log_file):
        results_dict = json.load(open(log_file, "r"))
    else:
        results_dict = {}

    group_train, group_test, group_val = get_datasets(args.dataset, args.group)
    df_all = pd.concat((group_train.df, group_test.df, group_val.df), axis=0, ignore_index=True)
    df_all = df_all.sample(frac=1)
    for model in args.model:
        print("Evaluating model {}".format(model))
        baseline_model = EntailmentCheckModel(model)
        times_list = []
        for i in range(args.n_retries):
            evidence = df_all.evidence.iloc[i*args.n_samples:(i+1)*args.n_samples]
            claims = df_all.claim.iloc[i*args.n_samples:(i+1)*args.n_samples]
            samples_list = list(zip(evidence, claims))
            t0 = time.time()
            baseline_model.compute_scores(samples_list, show_progress=True)
            t1 = time.time()
            total = t1 - t0
            times_list.append(total)
        results_dict[model] = times_list
        json.dump(results_dict, open(log_file, "w"))







