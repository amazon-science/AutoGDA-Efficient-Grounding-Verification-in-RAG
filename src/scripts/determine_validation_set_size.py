# A script performing finetuning on target data of different sized to find a suitable validation set size that
# is still insufficient for finetuning (otherwise it would make no sense to have the unsupervised approach)
import argparse
import json

## setup a test population
from src.utils.script_utils import get_datasets
from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.utils.script_utils import init_population_from_df
import os

if __name__ == "__main__":
    """
    Generates synthetic data from an existing dataset for premises. 
    For each premise it generates a hypothesis and a label from an LLM. 
    """
    os.environ["AWS_REGION_NAME"] = 'us-east-1'
    parser = argparse.ArgumentParser(prog='LLM api', description='', epilog='')
    parser.add_argument('-d', '--dataset', type=str, help="the dataset to use")
    parser.add_argument('-g', '--group', type=str, help="the subgroup of the datset.")
    parser.add_argument('-b', '--batchsize', type=int, default=2)
    parser.add_argument("-s", "--split", type=str, default="val", help="which dataset split to use for deriving the validation sets. Can be val or train.")
    parser.add_argument("-m", "--model", type=str, default="tasksource", help="the model to use")
    args = parser.parse_args()

    org_data_train, org_data_test, org_data_val = get_datasets(args.dataset, group=args.group)

    if args.split == "val":
        data_split_use = org_data_val
    elif args.split == "train":
        data_split_use = org_data_train
    else:
        raise ValueError("Split must be either 'val' or 'train'")

    all_evlist = list(data_split_use.df.evidence.unique())

    res_dict = {}
    if os.path.exists(f"results/val_set_size_{args.dataset}-{args.group}.json"):
        res_dict = json.load(open(f"results/val_set_size_{args.dataset}-{args.group}.json"))
    for num_use in [100]: # 1, 2, 4, 8, 16,
        if str(num_use) in res_dict or num_use in res_dict:
            continue
        res_dict[num_use] = {}
        for lr in [1e-3, 1e-4, 1e-5, 1e-6]: #[1e-3, 1e-4, 3e-4, 1e-5, 3e-5, 1e-6]
            evaluator = NLIFinetuningEvaluation(org_data_test, args.model, num_epochs=1, batch_size=2, lr=lr)
            df_subset = data_split_use.df[data_split_use.df.evidence.isin(all_evlist[:num_use])]
            popu_train = init_population_from_df(df_subset)
            res_dict[num_use][lr] = evaluator(popu_train, None)
        json.dump(res_dict, open(f"results/val_set_size_{args.dataset}-{args.group}.json", "w"))
