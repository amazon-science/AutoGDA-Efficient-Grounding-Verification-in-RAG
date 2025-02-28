## Implement utility functions
from src.sync_data.population import Population
import torch.nn as nn
import torch
from src.sync_data.compute_entailments import EntailmentCheckModel
from typing import Union
from tqdm import tqdm

class ModelLossUtility:
    """ Assigns points that have high loss a high utility. """
    def __init__(self, model: Union[EntailmentCheckModel, str] = "tasksource", multiplier=1.0):
        if isinstance(model, str):
            self.model = EntailmentCheckModel(model)
        else:
            self.model = model

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.multiplier = multiplier

    def __call__(self, popu: Population, tag) -> torch.Tensor:
        """ Forward the model on the samples and compute loss"""
        ## Extract sentence pairs and labels from population.
        sentence_pairs = list([[tag[0], item] for item in popu[tag]])
        labels = ([tag[1]]*len(sentence_pairs))
        labels = torch.tensor(labels, dtype=torch.long)
        S = torch.tensor(self.model.compute_scores(sentence_pairs, show_progress=False))
        S = torch.cat([1.0-S.reshape(-1, 1), S.reshape(-1, 1)], dim=1)
        loss = self.loss(S, labels)
        return self.multiplier*loss
