import typing as tp
import abc
import numpy as np
import torch
from transformers import AutoTokenizer
import pandas as pd
from copy import deepcopy

class Population():
    """ Maintain a collection of populations.
        The population maintains a collection of subpopulations, where subpopulations is identified by a tag that
        is non-mutable, e.g., int or a tuple. The tag is used to represent a tuple of (evidence / label) in our case.
        A specific subpopulation can be retrieved by indexing the Population with a tag, e.g., population[tag].
    """

    def __init__(self, target_population: tp.Union[None, tp.Dict[tp.Any, tp.Union[tp.List[str], np.ndarray]]] = None):
        if target_population is None:
            self.target_population = {}
        else:
            self.target_population = target_population
            for tag in self.target_population:
                if not isinstance(target_population[tag], np.ndarray):
                    self.target_population[tag] = np.array(self.target_population[tag])
        self.references = {}  # The back references indicating the parents of each sample
        self.agreement_frac = {}  # The agreement values between true labels and synthetic ones
        self.init_dist = {}


    def __len__(self):
        return sum(len(self[t]) for t in self.get_tags())

    def get_tags(self):
        return self.target_population.keys()

    def sample(self, n_samples: int, seed=None) -> "Population":
        """ Sample min(available, n_samples) items from each poputlation tag.
            Return new subpopulation.
        """
        subpopulation = Population()
        if seed is not None:
            state = np.random.RandomState(seed=seed)
        else:
            state = np.random.RandomState()
        for tag in self.get_tags():
            index_dict = state.permutation(len(self.target_population[tag]))[:n_samples]
            subpopulation[tag] = self[tag][index_dict]
            if self.get_references(tag) is not None:
                subpopulation.set_references(tag, self.get_references(tag)[index_dict])
            if self.get_agreement(tag) is not None:
                subpopulation.set_agreement(tag, self.get_agreement(tag)[index_dict])
            if self.get_initial_prob(tag) is not None:
                subpopulation.set_initial_prob(tag, self.get_initial_prob(tag)[index_dict])
        return subpopulation

    def get_indexed_subpopulation(self, index_dict):
        """ get a subpopulation with the specified indices.
            Index dict contains an np.ndarray for each tag.
        """
        subpopulation = Population()
        for tag in self.get_tags():
            if tag in index_dict:
                subpopulation[tag] = self[tag][index_dict[tag]]
                if self.get_references(tag) is not None:
                    subpopulation.set_references(tag, self.get_references(tag)[index_dict[tag]])
                if self.get_agreement(tag) is not None:
                    subpopulation.set_agreement(tag, self.get_agreement(tag)[index_dict[tag]])
                if self.get_initial_prob(tag) is not None:
                    subpopulation.set_initial_prob(tag, self.get_initial_prob(tag)[index_dict[tag]])
        return subpopulation


    def remove_duplicates(self):
        for tag in self.get_tags():
            self.target_population[tag] = np.unique(self.target_population[tag])

    def get_full_population(self) -> tp.Iterable[str]:
        return self.target_population

    def __getitem__(self, tag):
        return self.target_population[tag]

    def __setitem__(self, tag, value: tp.Union[tp.Tuple, tp.List[str], np.ndarray]):
        if isinstance(value, tuple):
            self.set_references(tag, value[1])
            if len(value) > 2: # also update pmiss
                self.set_agreement(tag, value[2])
            if len(value) > 3: # also set pinit
                self.set_initial_prob(tag, value[3])
            value = value[0]
        if not isinstance(value, np.ndarray):
            self.target_population[tag] = np.array(value)
        else:
            self.target_population[tag] = value

    def __add__(self, other):
        """ Join two populations. """
        print("join")
        for t in other.tags:
            if t in self.target_population:
                self.target_population[t] = np.concatenate((
                    self.target_population[t], other[t]))
            else:
                self.target_population[t] = other[t]
            if other.get_references(t) is not None:
                if self.get_references(t) is not None:
                    self.references[t] = np.concatenate((
                    self.references[t], other.get_references(t)))
                else:
                    self.references[t] = other.get_references(t)
            if other.get_initial_prob(t) is not None:
                if self.get_initial_prob(t) is not None:
                    self.init_dist[t] = np.concatenate((
                    self.init_dist[t], other.get_initial_prob(t)))
                else:
                    self.init_dist[t] = other.get_initial_prob(t)
            if other.get_agreement(t) is not None:
                if self.get_agreement(t) is not None:
                    self.agreement_frac[t] = np.concatenate((
                        self.agreement_frac[t], other.get_agreement(t)))
                else:
                    self.agreement_frac[t] = other.get_agreement(t)
        return self

    @property
    def tags(self):
        return self.get_tags()

    def get_total_size(self) -> int:
        return sum([len(self[g]) for g in self.get_tags()])

    def get_linear_list(self):
        """ Return a list of the instances in the population of all tags """
        return np.concatenate([self[g] for g in self.get_tags()])

    def get_vocab_per_tag(self, tokenizer_str):
        """ Return a dict where each tag is mapped to a tensor with a vocabulary. """
        t5_style = ("t5" in tokenizer_str)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        ret_dict = {}
        for tag in self.get_tags():
            ret = tokenizer(self[tag].tolist())["input_ids"]
            if t5_style:
                toks = torch.cat([torch.tensor(r[:-1], dtype=torch.long) for r in ret])
            else:
                toks = torch.cat([torch.tensor(r[1:-1], dtype=torch.long) for r in ret])
            ret_dict[tag] = toks.unique()
        return ret_dict

    def to_dataframe(self) -> pd.DataFrame:
        """ Build a pandas.DataFrame representation from the population. """
        dict_list = []
        for tag in self.get_tags():
            tag_dict = {}
            if isinstance(tag, tuple):
                for tag_idx, k in enumerate(tag):
                    tag_dict[f"tag_{tag_idx}"] = k
            else:
                tag_dict["tag"] = str(tag)
            for i, sample in enumerate(self[tag].tolist()):
                tmp_dict = deepcopy(tag_dict)
                tmp_dict["sample"] = sample
                dict_list.append(tmp_dict)
                if self.get_references(tag) is not None:
                    tmp_dict["ref"] = self.get_references(tag)[i]
                if self.get_agreement(tag) is not None:
                    tmp_dict["p_agree"] = self.get_agreement(tag)[i]
                if self.get_initial_prob(tag) is not None:
                    tmp_dict["p_init"] = self.get_initial_prob(tag)[i]
        return pd.DataFrame(dict_list)

    def reset_references(self):
        """ Make references point to itself. """
        for p in self.tags:
            self.set_references(p, np.arange(len(self[p])))

    def add_to_refs(self, constant: int):
        """ Add a constant to the references, which can be used for tracking the generation history of the samples. """
        for p in self.tags:
            myrefs = self.get_references(p)
            if myrefs is not None:
                self.set_references(p, myrefs+constant)

    def set_references(self, tag, refs):
        """ Set a list of references to a tag, they may refer e.g., to the parent samples in a previous population. """
        if isinstance(refs, np.ndarray):
            self.references[tag] = refs
        else:
            self.references[tag] = np.array(refs)

    def get_references(self, tag):
        if tag in self.references:
            return self.references[tag]
        else:
            return None

    def get_agreement(self, tag):
        if tag in self.agreement_frac:
            return self.agreement_frac[tag]
        else:
            return None

    def set_agreement(self, tag, refs):
        """ Set a list of references to a tag, they may refer e.g., to the parent samples in a previous population. """
        if isinstance(refs, np.ndarray):
            self.agreement_frac[tag] = refs
        else:
            self.agreement_frac[tag] = np.array(refs)

    def get_initial_prob(self, tag):
        if tag in self.init_dist:
            return self.init_dist[tag]
        else:
            return None

    def set_initial_prob(self, tag, refs):
        """ Set a list of references to a tag, they may refer e.g., to the parent samples in a previous population. """
        if isinstance(refs, np.ndarray):
            self.init_dist[tag] = refs
        else:
            self.init_dist[tag] = np.array(refs)

