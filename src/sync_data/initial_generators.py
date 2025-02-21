## Initial data generation routines.
import os

from src.utils.data_utils import AnnotatedTextDataset
from src.sync_data.compute_entailments import EntailmentCheckModel
from src.llm_sync_data.generate_summedits_sync import few_shot_prompting
from src.utils.script_utils import init_population_from_df, init_population_from_dump
from src.sync_data.population import Population
from tqdm import tqdm
import json
import pandas as pd
import numpy as np

## Synthetic data generators. Should be called with dataset after query integration.
class FewShotPromptingGenerator:
    def __init__(self, entailment_scoring_model: EntailmentCheckModel, model_name="claude3-haiku", min_req_examples=2,
                 prompt_mode = "summ", dataset="ragtruth", group="QA", openai_key_loc="data/openai.json"):
        """ prompt_mode: summ (for summaries) or qa for QA datasets. """
        self.entailment_scoring_model = entailment_scoring_model
        self.model_name = model_name
        self.min_examples_available = min_req_examples
        self.prompt_mode = prompt_mode
        self.dataset = dataset
        self.group = group
        oai_credentials = json.load(open(openai_key_loc))
        os.environ["OPENAI_API_KEY"] = oai_credentials["key"]

    def generate_samples(self, dataset_unsupervised: AnnotatedTextDataset, samples_per_evidence=4):
        """ Generate synthetic samples for each evidence using few shot prompting. """
        print("Generating samples...")
        sync_df = few_shot_prompting(dataset_unsupervised.df, n_generate=samples_per_evidence,
                                     n_few_shot=3, batch_size=4, model_name=self.model_name,
                                     min_req_examples=self.min_examples_available, prompt_mode="qa2" if self.group == "QA" else "summ")

        ## no perform entailment check using model.
        print("Computing initial entailments ...")
        ec_pairs = list(zip(sync_df.evidence, sync_df.claim))
        scores = self.entailment_scoring_model.compute_scores(ec_pairs)
        sync_df["p_init"] = scores
        return init_population_from_df(sync_df)

class FromFile:
    def __init__(self, type="ClaudeFewShot", path="sync_data", dataset="ragtruth", group="QA", generation_seed=1, select_max_cert=False, entail_model=None):
        """ Load the generated data from a file. """
        self.path = path
        self.type = type
        self.dataset = dataset
        self.group = group
        self.generation_seed = generation_seed
        self.select_max_cert = select_max_cert
        self.entail_model = entail_model

    def generate_samples(self, dataset_unsupervised: AnnotatedTextDataset, samples_per_evidence=4):
        """ Load generated samples from a file. """
        synth_data_train = pd.read_csv(f"{self.path}/{self.type}_{self.dataset}-{self.group}.csv")
        if self.select_max_cert:
            samples_per_evidence_use = 10000
        else:
            samples_per_evidence_use = samples_per_evidence

        if "tag_0" in synth_data_train:
            pop_init = init_population_from_dump(synth_data_train,  max_per_evidence=samples_per_evidence_use,
                                                 use_evidence=dataset_unsupervised.df.evidence.unique(),
                                                 seed=self.generation_seed)
        else:
            pop_init = init_population_from_df(synth_data_train,  max_per_evidence=samples_per_evidence_use,
                                               use_evidence=dataset_unsupervised.df.evidence.unique(),
                                               seed=self.generation_seed)
        if self.entail_model is not None: ## Recompute entailment scores.
            mynli = EntailmentCheckModel(self.entail_model)
            for t in tqdm(pop_init.tags, desc="Computing initial certainties"):
                sent_pairs = [[t[0], sentence] for sentence in pop_init[t]]
                scores = mynli.compute_scores(sent_pairs, show_progress=False)
                if t[1] == 0:
                    scores[scores > 0.5] = 0.5
                else: # t[1] ==1
                    scores[scores < 0.5] = 0.5
                pop_init.set_initial_prob(t, scores)

        if self.select_max_cert:
            return self._filter_max_cert(pop_init, samples_per_evidence)
        else:
            return pop_init

    def _filter_max_cert(self, population_in: Population, s_select):
        """ Filter the synthetic samples with the highest confidence score. """
        selection_dict = {}
        for t in population_in.tags:
            if t[1] == 0:
                selection_dict[t] = np.argsort(population_in.get_initial_prob(t))[:s_select]
            else: # t[1] == 1
                selection_dict[t] = np.argsort(-population_in.get_initial_prob(t))[:s_select]

        return population_in.get_indexed_subpopulation(selection_dict)