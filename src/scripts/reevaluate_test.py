## Perform evaluation on given run using weighted training objective.
import os
import sys
import pandas as pd
import json
from src.utils.script_utils import init_population_from_dump
from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.utils.script_utils import get_datasets
import argparse

def reevaluate(runfolder, model_list, org_data_test, useiters=[2], batchsize=2):
    if os.path.exists(f"{runfolder}/eval_out_weighted_test_list.json"):
        eval_weighed_dict_test = json.load(open(f"{runfolder}/eval_out_weighted_test_list.json"))
    else:
        eval_weighed_dict_test = {}
    if os.path.exists(f"{runfolder}/eval_out_unweighted_test_list.json"):
        myeval_unweight_dict_test = json.load(open(f"{runfolder}/eval_out_unweighted_test_list.json"))
    else:
        myeval_unweight_dict_test = {}

    for useiter in useiters:
        if str(useiter) not in myeval_unweight_dict_test:
            eval_weighed_dict_test[str(useiter)] = {}
            myeval_unweight_dict_test[str(useiter)] = {}
        for model in model_list:
            if model not in myeval_unweight_dict_test[str(useiter)]:
                print("Running eval on model: ", model, "iter=", useiter)
                #myeval_test = NLIFinetuningEvaluation(org_data_test, model, weighted=True, num_epochs=1,
                #                                      batch_size=batchsize)
                myeval_unweight_test = NLIFinetuningEvaluation(org_data_test, model, weighted=False, num_epochs=1,
                                                               batch_size=batchsize)
                if os.path.exists(f"{runfolder}/data_iteration_{useiter}.csv"):
                    mydf = pd.read_csv(f"{runfolder}/data_iteration_{useiter}.csv")
                    print("REEVAL of iteration:", useiter, "no samples:", len(mydf))
                    #eval_weighed_dict_test[str(useiter)][model] = myeval_test(init_population_from_dump(mydf), None)
                    #json.dump(eval_weighed_dict_test, open(f"{runfolder}/eval_out_weighted_test_list.json", "w"))
                    myeval_unweight_dict_test[str(useiter)][model] = myeval_unweight_test(init_population_from_dump(mydf), None)
                    json.dump(myeval_unweight_dict_test, open(f"{runfolder}/eval_out_unweighted_test_list.json", "w"))


if __name__ == '__main__':
    os.environ["AWS_REGION_NAME"] = 'us-east-1'
    parser = argparse.ArgumentParser(prog='LLM api', description='', epilog='')
    parser.add_argument('-d', '--dataset', type=str, help="the dataset to use")
    parser.add_argument('-g', '--group', type=str, help="the subgroup of the datset.")
    parser.add_argument("-f", "--run_folder", type=str)
    parser.add_argument('-b', '--batchsize', type=int, default=2)
    parser.add_argument("-m", "--model", type=str, default=None, nargs="+", help="which eval model to use")
    parser.add_argument("--lr", type=float, default=-1.0, help="The learning rate.")
    parser.add_argument("--useiters", type=int, nargs="+", default=[0], help="which iterations of the algorithm to use")
    args = parser.parse_args()
    print(args.run_folder)
    org_data_train, org_data_test, org_data_val = get_datasets(args.dataset, group=args.group)
    reevaluate(args.run_folder, args.model, org_data_test, useiters=args.useiters, batchsize=args.batchsize)

