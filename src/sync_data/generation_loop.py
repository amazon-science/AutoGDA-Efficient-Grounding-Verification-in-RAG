import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
#torch.multiprocessing.set_start_method('spawn')

import pandas as pd
import typing as tp
## Load initial data
from src.utils.data_utils import get_summedit_group_dataset
from src.sync_data.pc_mutations import LLMFillInTheGapsMutation
from src.sync_data.evaluators import NLIFinetuningEvaluation

from src.utils.script_utils import get_datasets, init_population_from_df, init_population_from_dump
from src.utils.data_utils import get_validation_evidence_size
from src.sync_data.construct_objects import get_data_generator, get_mutation, get_selector, get_utilites
## Initialize a population and compute embeddings
from src.sync_data.population import Population
import numpy as np
import json
import argparse
from src.sync_data.compute_entailments import EntailmentCheckModel
from src.sync_data.evaluators import PopulationWithEmbeddings
from src.sync_data.mutation_cache import MutationCache
from src.utils.script_utils import evaluate_dist_certainties, run_evaluators, create_topk_selection_dict
import warnings


def get_last_complete_iter(logpath):
    """ Get the number of the last complete iteraton for a run (used for restarting).
        Return a tuple of the iter index and the last population dump that was found (if the index is > 0)
    """
    if not os.path.exists(logpath):
        return 0
    check_iter = 1
    while os.path.isfile(f"{logpath}/data_iteration_{check_iter}.csv"):
        check_iter += 1
    last_complete_iter = check_iter - 1
    if last_complete_iter == 0:
        return last_complete_iter, None
    else:
        ## load dump
        mypop = init_population_from_dump(pd.read_csv(f"{logpath}/data_iteration_{last_complete_iter}.csv"))
        return last_complete_iter, mypop

def parse_args():
    parser = argparse.ArgumentParser(prog='Synthetic Data Generation', description='', epilog='')
    parser.add_argument("configfile", type=str, help="Path to config file. Mandatory.")
    parser.add_argument('-d', '--dataset', type=str, default='ragtruth')
    parser.add_argument('-g', '--group', type=str, default='QA', help='subgroup of the dataset')
    parser.add_argument('-r', '--run', type=str, default="opt_test")
    parser.add_argument('--evidences', type=int, default=-2, help="Number of evidences to use, -1 means use all.")
    parser.add_argument("--n_select", type=int, default=-1, help="how many samples to select per evidence/label tag.")
    parser.add_argument('--ft_batchsize', type=int, default=2, help='Batch size used for finetuning.')
    parser.add_argument('--eval_model', choices=['tasksource_v1', 'tasksource', 'vectara_v1', 'bart-large', "vectara_v2"], nargs="+")
    parser.add_argument('--num_iters', type=int, default=-1)
    parser.add_argument("--skip_init_eval", type=bool, default=False)
    args = parser.parse_args()
    return args


def unify_config(config, args):
    """ Overwrite config parameters by args from the command line.
        Negative argument values suggest using the parameter from the config.
    """
    if args.evidences > -2:
        config["evidences"] = args.evidences
    if args.n_select > 0:
        config["n_select"] = args.n_select
    config["ft_batchsize"] = args.ft_batchsize
    if args.eval_model is not None and len(args.eval_model) > 0:
        config["eval_model"] = args.eval_model
    if args.num_iters > 0:
        config["num_iters"] = args.num_iters
    config["skip_init_eval"] = args.skip_init_eval
    return config


def run_from_config(config, dataset: str, group: tp.Union[str, list[str]], run: str, save_path="runs"):
    """ Run the actual algorithm. All parameters besides dataset / group should be contained in the config argument.
        see config_files/base_config.json for an example.
    """
    
    ## Init OpenAI Keys
    if "open_ai_key_file" in config: 
        oai_credentials = json.load(open(config["open_ai_key_file"]))
        os.environ["OPENAI_API_KEY"] = oai_credentials["key"]

    ## Initialize and obtain processed datasets
    data_train, data_test, data_val = get_datasets(dataset, group)

    ## Perform evidence selection (for test runs, one might only use a subset of the evidence)
    list_evidence = data_train.df.evidence.unique()
    if config["evidences"] > 0:
        list_evidence = list_evidence[:config["evidences"]]

    data_train.df = data_train.df[data_train.df['evidence'].isin(list_evidence)]

    ## Set up validation set of correct size
    val_set_size = get_validation_evidence_size(dataset, group)
    all_evlist = list(data_val.df.evidence.unique())
    data_val.df = data_val.df[data_val.df.evidence.isin(all_evlist[:val_set_size])]
    print("Using validation set of size", len(data_val))

    ## Initialize generators
    print("Running initial data generation.")

    initial_population = Population()
    for generator_args in config["initial_generators"]:
        if "active" not in generator_args or generator_args["active"] is True:
            my_generator = get_data_generator(generator_args, base_dataset=dataset, group=group)
            generated_data_population = my_generator.generate_samples(data_train, generator_args["n_per_evidence"])
            initial_population = initial_population + generated_data_population
    print("Size of initial population:", len(initial_population))

    pop_target = init_population_from_df(data_train.df, labeled=False, use_evidence=list_evidence)
    print("Size of target population:", len(pop_target))

    ## Finetune a model on target and use that for claim entailment checking.
    test_ft_target = "test_ft_target" in config and config["test_ft_target"] is True
    target_ft_model = None
    if test_ft_target:
        target_ft_model = EntailmentCheckModel(config["eval_model"][0])
        target_ft_eval = NLIFinetuningEvaluation(data_val, target_model_str=target_ft_model,
                                                 num_epochs=config["ft_epochs"], device="cuda",
                                                 batch_size=config["ft_batchsize"], renew_model=False)
        print("Finetuning target model for (claim, claim) checking... ")
        target_ft_eval(init_population_from_df(data_train.df), None)  ## FT on target training data


    if "persistent_model" in config and config["persistent_model"] is True:
        # Use only one model that is continously updated in the loop
        # and which should be used for utility and checking of mutations
        if len(config["eval_model"]) > 1:
            raise ValueError("Cannot use more than one model when update_pmiss_model=True.")
        persistent_model = EntailmentCheckModel(config["eval_model"][0])
        persistent_model_mutation = persistent_model
        print("Initiated a persistent model of type:", config["eval_model"][0])
    else:
        persistent_model = None # indicates the model should be finetuned and used for uiltities.
        persistent_model_mutation = None # indicates the model used to check mutations

    if "single_rmiss" in config and config["single_rmiss"] is True:
        # This flag helps save GPU-RAM if all mutations use the same model for rmiss, it will be instantiated only once.
        # Note that this this is incompatible with "persistent_model" true.
        if persistent_model_mutation is not None:
            raise ValueError("Cannot use persistent model and single_rmiss model at the sampe time.")
        persistent_model_mutation = EntailmentCheckModel(config["mutations"][0]["entail_model"])

    ## Initialize Mutations from config
    mutation_list = []
    for mutation_args in config["mutations"]:
        if "active" not in mutation_args or mutation_args["active"] is True:
            if test_ft_target:
                mutation_obj = get_mutation(mutation_args, target_ft_model)
            else:
                mutation_obj = get_mutation(mutation_args, persistent_model_mutation)

            if config["mutation_cache"] is False:
                mutation_list.append(mutation_obj)
            else: # wrap in cache
                if isinstance(group, list):
                    group_name = "_".join(group)
                else:
                    group_name = group
                mutation_list.append(MutationCache(mutation_obj, dataset_name=f"{dataset}-{group_name}"))

    print(f"Using {len(mutation_list)} mutations.")

    # Computing Embeddings
    print("Computing initial embeddings.")
    target_w_emb = PopulationWithEmbeddings(pop_target)
    current_w_embed = PopulationWithEmbeddings(initial_population)

    ## Initialize the Selector
    if "utility" in config and config["utility"] is not None:
        utility_fn = get_utilites(config["utility"], persistent_model)
    else:
        utility_fn = None
    myselector = get_selector(config["selector"], target_w_emb, utility_fn=utility_fn)
    print(f"Constructed Selector.")

    evaluators_other = {}

    if persistent_model is None:
        for model_str in config["eval_model"]:
            print("evaluator: ", model_str)
            evaluators_other[model_str] = NLIFinetuningEvaluation(data_val, target_model_str=model_str, num_epochs=config["ft_epochs"], weighted=config["ft_weighted"],
                                                                  device="cuda", batch_size=config["ft_batchsize"])
    else:
        evaluators_other[config["eval_model"][0]] = NLIFinetuningEvaluation(data_val, target_model_str=persistent_model,
                                                              num_epochs=config["ft_epochs"], weighted=config["ft_weighted"],
                                                              device="cuda", batch_size=config["ft_batchsize"], renew_model=False)

    metric_list = {}
    objectives_list = {}
    eval_out_list = {}
    eval_in_list = {}
    log_path = os.path.join(save_path, run)
    os.makedirs(log_path, exist_ok=True)

    ## Restart?
    iter_start = 0
    if "restart" in config and config["restart"] == True:
        iter_start, pop_restart = get_last_complete_iter(log_path)
        if iter_start > 0:
            print(f"Restarting from iteration {iter_start}.")
            current_w_embed = PopulationWithEmbeddings(pop_restart)

    ## Main Loop
    for i in range(iter_start, config["num_iters"]):
        if os.path.exists(f"{log_path}/eval_in_list.json"):
            eval_in_list = json.load(open(f"{log_path}/eval_in_list.json"))
        if os.path.exists(f"{log_path}/eval_out_list.json"):
            eval_out_list = json.load(open(f"{log_path}/eval_out_list.json"))
        if os.path.exists(f"{log_path}/eval_objectives.json"):
            objectives_list = json.load(open(f"{log_path}/eval_objectives.json"))
        if os.path.exists(f"{log_path}/eval_metrics.json"):
            metric_list = json.load(open(f"{log_path}/eval_metrics.json"))

        if persistent_model is not None: # Reset persistent model before finetuning during evaluation is needed.
            persistent_model.reset_inplace()

        if i > iter_start or config["skip_init_eval"] is False:
            current_w_embed.to_dataframe().to_csv(f"{log_path}/data_iteration_{i}.csv", index=False)
            print(f"Evaluating on other evidences.")
            out_eval = run_evaluators(evaluators_other, current_w_embed)
            print(out_eval)
            eval_out_list[i] = out_eval
            #print(f"Evaluating on same evidences.")
            # in_eval = run_evaluators(evaluators_same, current_w_embed)
            # print(in_eval)
            # eval_in_list.append(in_eval)
            print(f"ITERATION {i}")
            print("Population size:", current_w_embed.get_total_size())

            ## Initial Eval (take care of log on restart)
            res_metrics = evaluate_dist_certainties(current_w_embed, target_w_emb)
            metric_list[i] = res_metrics
            for k, v in res_metrics.items():
                print(f"{k}: {np.array(v).mean()}")
            res_objective = myselector.evaluate_objective(current_w_embed)
            print(list(res_objective.values()))
            objectives_list[i] = list(res_objective.values())

            json.dump(eval_in_list, open(f"{log_path}/eval_in_list.json", "w"))
            json.dump(eval_out_list, open(f"{log_path}/eval_out_list.json", "w"))
            json.dump(objectives_list, open(f"{log_path}/eval_objectives.json", "w"))
            json.dump(metric_list, open(f"{log_path}/eval_metrics.json", "w"))

        elif persistent_model is not None:
            ## If we do not perform eval, we need to train the persistent model anyways
            print("Initial finetuning of NLI model... ")
            out_eval = run_evaluators(evaluators_other, current_w_embed)

        ## Mutation
        REF_CONSTANT = 100000
        pop_update_all = PopulationWithEmbeddings(Population())
        for num_m, m in enumerate(mutation_list):
            pop_update = PopulationWithEmbeddings(m.mutate_all_tags(current_w_embed))
            pop_update.add_to_refs((num_m+1)*REF_CONSTANT)
            pop_update_all = pop_update_all + pop_update
        res_metrics = evaluate_dist_certainties(pop_update_all, target_w_emb)

        metric_list[str(i)+"_mut"] = res_metrics
        for k, v in res_metrics.items():
            print(f"{k}: {np.array(v).mean()}")

        #Set references to self for current.
        current_w_embed.reset_references()
        current_w_embed = current_w_embed + pop_update_all
        ## Selection
        res_selection = myselector.select(current_w_embed, 16)
        res_selection_dict = create_topk_selection_dict(current_w_embed, res_selection, config["n_select"])
        current_w_embed = current_w_embed.get_indexed_subpopulation(res_selection_dict)

    ## FINAL EVAL
    print(f"Evaluating on other evidences.")
    out_eval = run_evaluators(evaluators_other, current_w_embed)
    print(out_eval)
    eval_out_list[config["num_iters"]] = out_eval
    #print(f"Evaluating on same evidences.")
    # in_eval = run_evaluators(evaluators_same, current_w_embed)
    # print(in_eval)
    # eval_in_list.append(in_eval)
    print(f"ITERATION {config['num_iters']}")
    print("Population size:", current_w_embed.get_total_size())
    ## Initial Eval
    #res_metrics = evaluate_dist_certainties(current_w_embed, target_w_emb)
    #metric_list[config["num_iters"]] = res_metrics

    #res_objective = myselector.evaluate_objective(current_w_embed)
    #print(list(res_objective.values()))
    #objectives_list[config["num_iters"]] = list(res_objective.values())

    json.dump(eval_in_list, open(f"{log_path}/eval_in_list.json", "w"))
    json.dump(eval_out_list, open(f"{log_path}/eval_out_list.json", "w"))
    json.dump(metric_list, open(f"{log_path}/eval_metrics.json", "w"))
    json.dump(objectives_list, open(f"{log_path}/eval_objectives.json", "w"))
    current_w_embed.to_dataframe().to_csv(f"{log_path}/data_iteration_{config['num_iters']}.csv", index=False)

    ## Final eval.
    if "do_final_eval" in config and config["do_final_eval"] is True:
        print("Running final evaluation on testset. ")
        if persistent_model is None:
            for model_str in config["eval_model"]:
                evaluators_other[model_str] = NLIFinetuningEvaluation(data_test, target_model_str=model_str,
                                                                      num_epochs=config["ft_epochs"], weighted=config["ft_weighted"],
                                                                      device="cuda", batch_size=config["ft_batchsize"])
        else:
            evaluators_other[config["eval_model"][0]] = NLIFinetuningEvaluation(data_test, target_model_str=persistent_model,
                                                                  num_epochs=config["ft_epochs"], weighted=config["ft_weighted"],
                                                                  device="cuda", batch_size=config["ft_batchsize"])
        out_eval = run_evaluators(evaluators_other, current_w_embed)
        print("Final:", out_eval)
        json.dump(out_eval, open(f"{log_path}/eval_out_final.json", "w"))

    return eval_out_list[config["num_iters"]][config["eval_model"][0]] ## Return last eval on val set


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    config = json.load(open(args.configfile))
    config = unify_config(config, args)
    run_from_config(config, args.dataset, args.group, args.run)
    evaluators_other = {}








