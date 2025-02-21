## Perform evaluation on given run using weighted training objective.
import os
import sys
import pandas as pd
import json
from src.utils.script_utils import init_population_from_dump
from src.sync_data.evaluators import dataset_from_population
from src.sync_data.compute_entailments import EntailmentCheckModel
from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.utils.script_utils import get_datasets
from src.utils.data_utils import perform_query_integration

org_data_train, org_data_test, org_data_val = get_datasets("ragtruth", group="QA")
querymode = "queryonly"
org_data_test.df = perform_query_integration(org_data_test.df, mode=querymode)
org_data_val.df = perform_query_integration(org_data_val.df, mode=querymode)

mymodel_persistent = EntailmentCheckModel(sys.argv[1])
myeval = NLIFinetuningEvaluation(org_data_val, mymodel_persistent, weighted=True, num_epochs=1, batch_size=2)
myeval_test = NLIFinetuningEvaluation(org_data_test, mymodel_persistent, weighted=True, num_epochs=0, batch_size=2)
myeval_unweight_test = NLIFinetuningEvaluation(org_data_test, sys.argv[1], weighted=False, num_epochs=1, batch_size=2)
for run in sys.argv[2:]:
    if not os.path.isdir("runs/"+run) or not run.startswith("ablation"):
        continue
    print(f"run={run}")
    eval_weighed_dict = {}
    eval_weighed_dict_test = {}
    myeval_unweight_dict_test = {}
    for useiter in [2]:
        print("REEVAL of iteration:", useiter)
        if os.path.exists(f"runs/{run}/data_iteration_{useiter}.csv"):
            mymodel_persistent.reset_inplace()
            mydf = pd.read_csv(f"runs/{run}/data_iteration_{useiter}.csv")
            print("Found samples:", len(mydf))
            # dataset = dataset_from_population(init_population_from_dump(mydf))
            eval_weighed_dict[useiter] = myeval(init_population_from_dump(mydf), None)
            json.dump(eval_weighed_dict, open(f"runs/{run}/eval_out_weighted_list.json", "w"))
            eval_weighed_dict_test[useiter] = myeval_test(init_population_from_dump(mydf), None)
            json.dump(eval_weighed_dict_test, open(f"runs/{run}/eval_out_weighted_test_list.json", "w"))
            myeval_unweight_dict_test[useiter] = myeval_unweight_test(init_population_from_dump(mydf), None)
            json.dump(myeval_unweight_dict_test, open(f"runs/{run}/eval_out_unweighted_test_list.json", "w"))
