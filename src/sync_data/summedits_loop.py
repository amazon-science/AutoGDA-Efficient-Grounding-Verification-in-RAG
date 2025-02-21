import pandas as pd
from src.utils.data_utils import get_summedit_dataset
from src.sync_data.population import Population
from src.sync_data.mutation import LLMFillInTheGapsMutation
from src.sync_data.mutation import LLMTargetedFillInTheGaps
from src.sync_data.evaluators import VectaraFinetuningEvaluation
from src.sync_data.filtering import NLIModelEntailmentCheck
from copy import deepcopy
from synth_data_utils import store_csvdata_local_and_s3
import json
from augmented_evolution import AugmentedEvolutionSelector
import sys

if len(sys.argv) < 2:
    print("please pass a summedits subset.")
    exit(-1)
else:
    subset = sys.argv[1]
    s3bucket ="tleemann-store"

run_id = "test-select"
n_sample = 5
n_keep = 15


data = get_summedit_dataset("full")
datadf = data.df
datadf_bill = datadf[datadf["group"] == subset]



# Create initial population
org_samples = datadf_bill[datadf_bill.id.str.contains("og")]
population_dict0, population_dict1, population_dict_target = {}, {}, {}
population_wordlist = {}
for rowid, row0 in org_samples.iterrows():
    population_dict0[(row0.evidence, 0)] = [row0.claim]
    population_dict1[(row0.evidence, 1)] = [row0.claim]
    # Target vocab
    use_evidence = row0.evidence
    claims = datadf_bill[datadf_bill.evidence==use_evidence].claim.unique()
    population_dict_target[(row0.evidence, 0)] = claims
    population_dict_target[(row0.evidence, 1)] = claims
    wordlist = []
    for c in claims:
        wordlist.extend(c.split(" "))
    wordlist = set(wordlist)
    population_wordlist[(use_evidence, 0)] = wordlist
    population_wordlist[(use_evidence, 1)] = wordlist

mypop0 = Population(population_dict0)
mypop1 = Population(population_dict1)
mytarget = Population(population_dict_target)
my_selector = AugmentedEvolutionSelector(target_population=mytarget, consider_all_neighbors=True, select_unique=True, strategy="topk")

mutate_targeted = LLMTargetedFillInTheGaps(word_list=population_wordlist, n_mutations=5, n_multiply=2, mask_output_perc=0.2)
mutate_untargeted = LLMFillInTheGapsMutation(n_mutations=5, n_multiply=2, preserve_meaning=True, mask_output_perc=0.2)
res_list = []

myeval = VectaraFinetuningEvaluation(f"summedits-{subset}/test", num_epochs=3)
pop_init = Population()
pop_init += mypop0
pop_init += mypop1
res = myeval(pop_init, None)
print(res)
res_list.append(res)
for gen in range(5):
    mypop0mutate = mutate_targeted.mutate_all_tags(mypop0)
    mypop1mutate = mutate_untargeted.mutate_all_tags(mypop1)

    ## Test with n_sample
    test_pop0 = Population(mypop0mutate.sample(n_sample))
    test_pop1 = Population(mypop1mutate.sample(n_sample))
    test_pop0 = test_pop0 + test_pop1
    res = myeval(test_pop0, None)
    store_csvdata_local_and_s3(test_pop0.to_dataframe(), s3bucket,
                               f"runs_summ/{subset}/data_step_{gen}_mutate.csv")

    print("After mutate", res)
    res_list.append(res)
    ## Selection step
    filter_fn_0 = NLIModelEntailmentCheck(target_model_local_path="vectara/hallucination_evaluation_model",
                                          filter_higher=False, threshold=0.6, min_keep=n_keep)
    filter_fn_1 = NLIModelEntailmentCheck(target_model_local_path="vectara/hallucination_evaluation_model",
                                          filter_higher=True, threshold=0.996, min_keep=n_keep)
    pfilter0 = filter_fn_0.filter_entailment_scores(deepcopy(mypop0mutate), mypop0)
    pfilter1 = filter_fn_1.filter_entailment_scores(deepcopy(mypop1mutate), mypop1)

    test_pop0 = Population(pfilter0.sample(n_sample))
    test_pop1 = Population(pfilter1.sample(n_sample))

    test_pop0 = test_pop0 + test_pop1
    res = myeval(test_pop0, None)
    print("After filter", res)
    store_csvdata_local_and_s3(test_pop0.to_dataframe(), s3bucket,
                               f"runs_summ/{run_id}/{subset}/data_step_{gen}_filter.csv")
    res_list.append(res)
    json.dump(res_list, open(f"runs_summ/{run_id}/{subset}.json", "w"))


    mypop0 += pfilter0
    #mypop0 = Population(mypop0.sample(15))
    mypop0 = my_selector.select(mypop0, num_select=n_keep)
    mypop1 += pfilter1
    mypop1 = my_selector.select(mypop1, num_select=n_keep)
    #mypop1 = Population(mypop1.sample(15))

    print("Size after sampling:", len(mypop0), len(mypop1))



