## Cache generated samples in logfiles for faster iterations.
from src.sync_data.population import Population
from src.sync_data.pc_mutations import ProbCorrectMutation
import numpy as np
import os
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import torch
import time
from pandas.api.types import is_float_dtype

class MutationCache:
    def __init__(self, base_mutation: ProbCorrectMutation, dataset_name: str):
        self.run_name = dataset_name
        self.base_mutation = base_mutation
        self.mutation_str = type(base_mutation).__name__
        self.cache_path = f"runs/cache/{dataset_name}/{self.mutation_str}.csv"
        self.cache = None
        self.n_target = self.base_mutation.num_per_sample()  # how many mutations per input
        if os.path.exists(self.cache_path):
            self.cache = pd.read_csv(self.cache_path)
            self.cache["used"] = False

    def update_cache(self, population_org, population_update, rmiss_dict):
        """ Add newly generated samples to cache and store cache on disk.
        """
        dict_list = []
        for tag in population_update.get_tags():
            tag_dict = {"tag_0": tag[0], "tag_1": tag[1]}
            refs = population_update.get_references(tag)
            for i, sample in enumerate(population_update[tag].tolist()):
                tmp_dict = deepcopy(tag_dict)
                tmp_dict["old_claim"] = population_org[tag][refs[i]]
                tmp_dict["new_claim"] = sample
                tmp_dict["used"] = True
                if self.base_mutation.entail_model_identifier != "custom":
                    tmp_dict[f"rmiss_{self.base_mutation.entail_model_identifier}"] = rmiss_dict[tag][i]
                dict_list.append(tmp_dict)
        if len(dict_list) == 0: ## Empty population
            return
        updates = pd.DataFrame(dict_list)
        if os.path.exists(self.cache_path):
            prior_cache = pd.read_csv(self.cache_path)
            write_cache = pd.concat([prior_cache, updates], axis=0, ignore_index=True)
            ## Merge while keeping entailment values for all models...
            write_cache = write_cache.groupby(["tag_0", "tag_1", "old_claim", "new_claim"]).mean().reset_index()
            write_cache["used"] = True
            ## Sanity check of write cache
            for row in write_cache.columns:
                if row.startswith("rmiss"):
                    assert is_float_dtype(write_cache[row])
        else:
            updates = updates.drop_duplicates(["tag_0", "tag_1", "old_claim", "new_claim"])
            write_cache = updates
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        ## Fuse potential updates
        write_cache.to_csv(self.cache_path, index=False)

    def mutate_all_tags(self, input_population: Population):
        ## find all cache hits

        cache_output_population = None
        remain_dict = None
        if self.cache is not None:
            cache_output_population = Population()
            remain_dict = {}
            pmiss_update_dict = {}
            batch_time_list =[]
            cache_update_required = False
            for t in tqdm(input_population.tags, desc="Scanning cache for mutations"):
                sub_df = (self.cache.tag_0 == t[0]) & (self.cache.tag_1 == t[1])
                sample_list = [] # List of discovered samples from cache
                p_id_list = [] # List of parent ids
                remain_list = [] # List of remaining parent ids
                rmiss_list = [] # list of loaded rmiss scores or NaN if they are not available.
                for p_id, claim in enumerate(input_population[t]):
                    index = (sub_df & (self.cache.old_claim == claim) & (self.cache.used == False))
                    if np.sum(index.values) >= self.n_target: ## samples available in the cache
                        relevant_items = self.cache.loc[self.cache.index[index][:self.n_target]]
                        sample_list.append(relevant_items.new_claim.values)
                        if f"rmiss_{self.base_mutation.entail_model_identifier}" in self.cache.columns:
                            rmiss_list.append(relevant_items[f"rmiss_{self.base_mutation.entail_model_identifier}"].values)
                            #print("Samples w/o rmiss:", np.sum(np.isnan(rmiss_list)))
                        else:
                            rmiss_list.append(np.ones(len(relevant_items))*float("nan"))
                        self.cache.loc[self.cache.index[index][:self.n_target], "used"] = True
                        p_id_list.append(np.ones(self.n_target, dtype=np.int64)*p_id)
                    else:
                        remain_list.append(p_id) # cache miss
                if len(p_id_list) > 0: # Samples from the cache are used, compute their rmiss if required.
                    p_id_list = np.concatenate(p_id_list)
                    sample_list = np.concatenate(sample_list)
                    rmiss_list = np.concatenate(rmiss_list)
                    t0 = time.time()
                    if self.base_mutation.nli_model is not None:
                        pmisslabel = rmiss_list
                        if np.isnan(pmisslabel).any():
                            pmisslabel[np.isnan(rmiss_list)] = self.base_mutation._compute_rmiss(input_population[t].tolist(), sample_list[np.isnan(rmiss_list)], p_id_list[np.isnan(rmiss_list)], tag=t)
                            cache_update_required = True
                    else:
                        pmisslabel = np.ones(len(sample_list)) * self.base_mutation.miss_prob
                    pmiss_update_dict[t] = pmisslabel
                    t1 = time.time()
                    batch_time_list.append(t1 - t0)
                    if self.base_mutation.check_against == "parent":
                        p_org = input_population.get_initial_prob(t)
                        p_org_update = p_org[p_id_list]
                        pagree_prior = input_population.get_agreement(t)[p_id_list]  # Update misslabel probabilities.
                        pagree_update = (1.0 - pmisslabel) * pagree_prior + pmisslabel * (1.0 - pagree_prior)
                    else:
                        p_org_update = pmisslabel
                        pagree_update = np.ones(len(p_org_update))
                    cache_output_population[t] = sample_list, p_id_list, pagree_update, p_org_update
                if len(remain_list) > 0:
                    remain_dict[t] = np.array(remain_list)  # Cache misses, still to do.
            print("Cache hits: ", len(cache_output_population))
            ## Writeback pmiss values
            if cache_update_required:
                self.update_cache(input_population, cache_output_population, pmiss_update_dict)
            if len(batch_time_list) > 0:
                print("Average tag time: ", np.array(batch_time_list).mean())
            population_todo = input_population.get_indexed_subpopulation(remain_dict)
            print("Remaining examples not in cache: ", len(population_todo))
        else:
            population_todo = input_population

        ## create new samples.
        if len(population_todo) > 0:
            population_update, rmiss_dict = self.base_mutation.mutate_all_tags(population_todo, return_raw_rmiss=True)
            self.update_cache(population_todo, population_update, rmiss_dict)
        else:
            population_update = Population()

        ## fix references
        if remain_dict is not None:
            for t in remain_dict:
                population_update.references[t] = remain_dict[t][population_update.get_references(t)]
        ## Merge populations
        if cache_output_population is not None:
            population_update = population_update + cache_output_population

        return population_update
