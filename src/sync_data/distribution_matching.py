import sys
import os
sys.path.append('../')
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from src.utils.data_utils import AnnotatedTextDataset, DATA_LOADER_FN
from src.sync_data.population import Population
from src.sync_data.augmented_evolution import AugmentedEvolutionSelector
from src.sync_data.mutation import (WordExtensionMutation, WordDeletionMutation, RephraseMutation,
                                    WordReplacementMutation, MutationList, APIRephrasingMutation, APIGrammarCorrection,
                                    LLMFillInTheGapsMutation)
from evaluators import (PopulationWithEmbeddings, NNCoverageEvaluator, PerplexityEvaluation,
                        add_summary_stats, dict_tags_to_str, FrechetInceptionDistance, ManifoldClassification,
                        VectaraInferenceEvaluation, VectaraFinetuningEvaluation)

from synth_data_utils import store_json_local_and_s3, store_csvdata_local_and_s3
import hashlib
from copy import deepcopy
import argparse
import logging
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Distribution Matching', description='', epilog='')
    parser.add_argument('-m', '--model', default='vectara/hallucination_evaluation_model')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--n_target", type=int, default=1000)
    parser.add_argument("-s", "--dataset_source", type=str, default='summedits-billsum/syncv2haiku')
    parser.add_argument("-t", "--dataset_target", type=str, default='summedits-billsum/train')
    parser.add_argument("-p", "--population_size", type=int, default=6)
    parser.add_argument("-o", "--n_offspring", type=int, default=8, help="Number of offspring to generate in mutation step.")
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--n_iter", type=int, default=40)
    parser.add_argument("-r", "--run_name", type=str, default="test", help="A name of run. The results will be stored in a folder with this name.")
    parser.add_argument("--s3bucket", type=str, default='tleemann-store')
    parser.add_argument("--restart_iter", type=int, default=0, help="will restart the algorithm using logged data from a previous iteration.")

    args = parser.parse_args()
    print(args)
    import os

    model_name = args.model
    device = args.device

    source_dataset = DATA_LOADER_FN[args.dataset_source]()
    target_dataset = DATA_LOADER_FN[args.dataset_target]()

    ### TODO: Clean this up, this is for testing purposes as of now.
    # match by claims.
    dfselect_list_target = []
    dfselect_list_source = []
    print("=== Source Claims ===")
    # Dictionaries of claims.
    target_dict = {}
    source_dict = {}
    for i in range(min(args.n_target, len(target_dataset))):
        ## identify source claims
        target_evidence = target_dataset.df.iloc[i]["evidence"]
        #source_evidence = source_dataset.df.iloc[i]["evidence"]
        df_select_src = source_dataset.df[source_dataset.df["evidence"] == target_evidence]
        ## print a faithful seed summary
        df_select_tar = target_dataset.df[target_dataset.df["evidence"] == target_evidence]
        for label in [0, 1]:
            target_dict[(target_evidence, label)] = (df_select_tar[df_select_tar["label_binary"] == label]["claim"]).values
            source_dict[(target_evidence, label)] = (df_select_src[df_select_src["label_binary"] == label]["claim"]).values

    if args.restart_iter > 0: ## Restart with a different target population
        my_population = pd.read_csv(f"runs/{args.run_name}/data_step_{args.restart_iter}.csv")
        for key in source_dict.keys():
            target_evidence, label = key
            index = (my_population["tag_0"] == target_evidence) & (my_population["tag_1"] == label)
            source_dict[(target_evidence, label)] = my_population[index]["sample"].values


    #source_dict["test"] = ["This manuscript was completely unrelated to my research."]*15
    #target_dict["test"] = ["Paris has become a popular international tourist destination."]
    #for t in target_dict.keys(): # just one item
    #    target_dict[t] = [target_dict[t][0]]

    debug_tag = list(target_dict.keys())[0]
    print("Target:", target_dict[debug_tag])

    target_population = Population(target_dict)
    source_population = Population(source_dict)

    # Genetic Algorithm implementation
    print(f"Size of initial population: {source_population.get_total_size()},",
          f"Size of target population {target_population.get_total_size()}.")

    target_vocab = target_population.get_vocab_per_tag(tokenizer_str="google-t5/t5-base")
    metric_list = [NNCoverageEvaluator(normalize=False),
                   PerplexityEvaluation(model_str="google-t5/t5-base", tokenizer_str="google-t5/t5-base"),
                   ]

    # List of metrics
    vectara_inference_eval = VectaraInferenceEvaluation("checkpoints/crosstest_billsum")
    vectara_finetune_eval = VectaraFinetuningEvaluation("summedits-billsum/train", num_epochs=3)

    vectara_corrector = APIGrammarCorrection(n_mutations=1, n_threads=4)
    my_selector = AugmentedEvolutionSelector(target_population, strategy="topk", histogram=False,
                                             model_str=args.embedding_model)

    mutation_list = LLMFillInTheGapsMutation(mask_output_perc=0.25, temperature=1.0, n_multiply=5, n_threads=4, preserve_meaning=True)

    #WordReplacementMutation(n_mutations=10, model_str="google-t5/t5-base",
    #                        tokenizer_str="google-t5/t5-base", consider_top_k=300, n_multiply=multiply),
    #WordReplacementMutation(n_mutations=5, model_str="google-t5/t5-base", n_multiply=multiply,
    #                        tokenizer_str="google-t5/t5-base", consider_top_k=10, use_vocab=target_vocab),
    #WordExtensionMutation(n_mutations=5, model_str="google-t5/t5-base", n_multiply=multiply,
    #                        tokenizer_str="google-t5/t5-base", consider_top_k=200),
    #WordExtensionMutation(n_mutations=5, model_str="google-t5/t5-base", n_multiply=multiply,
    #                        tokenizer_str="google-t5/t5-base", consider_top_k=10, use_vocab=target_vocab),
    #WordDeletionMutation(n_mutations=8, tokenizer_str="google-t5/t5-base"),
    #APIRephrasingMutation(n_mutations=3),
    #APIGrammarCorrection(n_mutations=5)
    #])
    # logging.basicConfig(level=logging.INFO)
    #mutation_list.extend([RephraseMutation(max_length=35, n_mutations=2)])
    #print("Number of Mutations: ", len(mutation_list))
    current_population = source_population
    initial_eval = VectaraInferenceEvaluation("checkpoints/crosstest_billsum", eval_target=True)

    if args.restart_iter > 0:
        metrics = json.load(open(f"runs/{args.run_name}/metrics.json"))
    else:
        metrics = {}

    ## DEBUG CODE
    eval_single_res = dict_tags_to_str(initial_eval(current_population, target_population))
    metrics["-1"] = eval_single_res

    tp_with_embs = PopulationWithEmbeddings(target_population, embedding_model_str=args.embedding_model)

    if args.restart_iter > 0: ## Start with the mutation step in case of restart, because it would be next after dump.
        current_population = mutation_list.mutate_all_tags(current_population)
        current_population.remove_duplicates()
        print("Size of current population:", current_population.get_total_size())

    for generation in range(args.restart_iter, args.n_iter):
        print(f" ========= ITERATION {generation} ========= ")
        # Selection step: Select args.population_size neighbors for each sample
        current_population = my_selector.select(current_population, args.population_size)
        current_population.remove_duplicates()
        print(current_population[debug_tag][:3])

        ## Evaluation step.
        cp_with_embs = PopulationWithEmbeddings(current_population, embedding_model_str=args.embedding_model)
        res_eval = {}
        for evaluation in metric_list:
            eval_single_res = evaluation(cp_with_embs, tp_with_embs)
            res_eval.update(eval_single_res)

        corrected_population = vectara_corrector.mutate_all_tags(deepcopy(current_population))
        res_eval.update(vectara_inference_eval(corrected_population, target_population))
        res_eval.update(vectara_finetune_eval(corrected_population, target_population))
        metrics[generation] = add_summary_stats(dict_tags_to_str(res_eval))

        ## Store metrics and samples locally and in S3
        for m in metrics[generation].keys():
            if "avg" in m:
                print(f"{m}: {metrics[generation][m]}")
        store_json_local_and_s3(metrics, args.s3bucket, f"runs/{args.run_name}/metrics.json")
        store_csvdata_local_and_s3(current_population.to_dataframe(), args.s3bucket,
                                   f"runs/{args.run_name}/data_step_{generation}.csv")


        # Mutation step.
        current_population = mutation_list.mutate_all_tags(current_population)
        current_population.remove_duplicates()
        print("Size of current population:", current_population.get_total_size())
        # compute list of metrics



    ## TODO: Proper saving of outputs and results
    print("====== Result: ")
    with open("dump_samples.txt", "w") as outputfile:
        for pi in current_population.get_linear_list():
            outputfile.write(f"{pi}\n")

