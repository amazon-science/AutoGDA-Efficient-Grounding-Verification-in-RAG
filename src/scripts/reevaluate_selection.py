## Perform evaluation on given run using weighted training objective.
import os
import sys
import pandas as pd
import json
import numpy as np


from src.utils.script_utils import init_population_from_dump
from src.sync_data.evaluators import dataset_from_population
from src.sync_data.compute_entailments import EntailmentCheckModel
from src.sync_data.evaluators import NLIFinetuningEvaluation
from src.utils.script_utils import get_datasets
from src.utils.data_utils import perform_query_integration
from src.utils.script_utils import init_population_from_dump
from src.sync_data.population import Population

org_data_train, org_data_test, org_data_val = get_datasets("ragtruth", group="QA")
querymode = "queryonly"
org_data_test.df = perform_query_integration(org_data_test.df, mode=querymode)
org_data_val.df = perform_query_integration(org_data_val.df, mode=querymode)

mymodel_persistent = EntailmentCheckModel("tasksource")
myeval = NLIFinetuningEvaluation(org_data_val, mymodel_persistent, weighted=False, num_epochs=1, batch_size=2)
myeval_test = NLIFinetuningEvaluation(org_data_test, mymodel_persistent, weighted=False, num_epochs=0, batch_size=2)
eval_selection ={}
for run in [sys.argv[1]]:
    res_objectives = json.load(open(f"runs/{run}/eval_objectives.json"))
    objective_values = np.array(list(res_objectives.values()))
    argmins = np.argmin(objective_values, axis=0)
    population = Population()
    myres = {}
    for i in range(4):
        sync_data = pd.read_csv(f"runs/{run}/data_iteration_{i}.csv")
        myres[i] = init_population_from_dump(sync_data)
        print(len(myres[i]))
    for i, t in enumerate(myres[0].tags):
        population[t] = myres[argmins[i]][t]
    eval_selection["val"] = myeval(population, None)
    eval_selection["test"] = myeval_test(population, None)
    json.dump(eval_selection, open(f"runs/{run}/eval_out_selection.json", "w"))