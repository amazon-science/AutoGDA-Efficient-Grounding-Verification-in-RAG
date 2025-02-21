## implementation of the Augmented PE algorithm from Xie et al., ICML 2024
import abc
import faiss
from collections import Counter

from sentence_transformers import SentenceTransformer
import torch
from abc import ABC
import typing as tp
from tqdm import tqdm
import numpy as np
from src.sync_data.population import Population

class OffspringSelector(ABC):
    """ Interface for the the offspring selection strategy.
        Given a set of samples, we select a subset, that should be further mutated.
        The number of samples that should be selected can be passed as an argument to the select function.
    """
    def __init__(self, target_population: Population):
        self.target_population = target_population

    @abc.abstractmethod
    def select(self, population: tp.List[str], num_select: int) -> tp.List[str]:
        raise NotImplementedError("Please Implement the select method in a subclass of OffspringSelector.")

    def __call__(self, population: tp.List[str], num_select: int) -> tp.List[str]:
        return self.select(population, num_select)


class AugmentedEvolutionSelector(OffspringSelector):
    """ Implementation of the AugmentedPE selection strategy (Xie et al.)"""

    def __init__(self, target_population: Population,
                 model_str ="sentence-t5-base",
                 device="cuda", batch_size=1000, strategy = "sampling", histogram=True,
                 temperature=0.01, consider_all_neighbors=True, select_unique=False) -> None:
        """
        :param: strategy: either 'sampling' or 'topk'.
        Sampling will sample the offspring samples according to the histogram, 'topk' will choose the most probable samples.
        'topk' does not repeat samples, but it is also not entirely faithful to the distribution mandated by the histogram.
        :param: histogram: If true use histogram, if false the sampling weights will be simply estimated
        through a nearest neighbor computation.
        :param: consider_all_neighbors: The topk neighbors for EACH target sample will be considered, not only the neareast one. This will induce more diversity, but th
        :param: select_unique: If true only select unique samples, if false, the same samples can be selected different times.
        """
        super(AugmentedEvolutionSelector, self).__init__(target_population)
        self.model_str = model_str
        self.device = device
        self.batch_size = batch_size
        self.strategy = strategy
        self.histogram = histogram
        self.temperature = temperature
        self.consider_all = consider_all_neighbors
        self.select_unique = select_unique
    def select(self, population: Population, num_select: int) -> Population:
        """
            Perform selection.
            num_select: How many samples to select per Tag.
        """
        # compute embeddings for both populations.
        for t in tqdm(population.tags, total=len(population.tags)):
            target_embeddings = torch.from_numpy(self._extract_features(self.target_population[t]))
            population_embeddings = torch.from_numpy(self._extract_features(population[t]))
            #dist_mat = torch.cdist(target_embeddings, population_embeddings)

            if self.histogram:
                counts = self._nn_histogram(population_embeddings, target_embeddings)
                counts = counts / counts.sum()
            else: ## Nearest neighbor
                counts = torch.cdist(target_embeddings, population_embeddings)
                if not self.consider_all:
                    counts = counts.min(dim=0)[0]
                counts = torch.softmax(-counts/self.temperature, dim=-1)

            #print("Mean embedding distance: ", dist_mat.mean(), "non-zero: ", torch.sum(counts > 0))
            if self.strategy == "sampling":
                sel_idx = torch.multinomial(counts, num_samples=num_select, replacement=True)
            elif self.strategy == "topk":
                values, sel_idx = torch.topk(counts, k=min(num_select, len(population_embeddings)))
            else:
                raise ValueError(f"Unknown selection strategy {self.strategy}.")
            if self.consider_all:
                sel_idx = sel_idx.t().flatten()
            if self.select_unique:
                sel_idx = torch.unique_consecutive(sel_idx)[:num_select]
                population[t] = list(population[t][sel_idx.numpy()])
            else:
                population[t] = list(population[t][sel_idx.numpy()])
        return population

    def _extract_features(self, data):
        # If available, the model is automatically executed on the GPU.
        model = SentenceTransformer(self.model_str, device=self.device)  # ='cuda',
        model.eval()
        batch_size = self.batch_size
        with torch.no_grad():
            sentence_embeddings = []
            for i in range(len(data) // batch_size+1):
                embeddings = model.encode(
                    data[i * batch_size:(i + 1) * batch_size])
                if len(embeddings) > 0:
                    sentence_embeddings.append(embeddings)
        sentence_embeddings = np.concatenate(sentence_embeddings)
        del model
        return sentence_embeddings

    def _nn_histogram(self, public_features, private_features,
                        num_packing=1, num_nearest_neighbor=1, mode='L2'):

        """ Compute the histogram of the nearest neighbors. Code adapted
        from https://github.com/AI-secure/aug-pe/blob/main/dpsda/dp_counter.py.

         """
        assert public_features.shape[0] % num_packing == 0
        num_true_public_features = public_features.shape[0] // num_packing
        if public_features.shape[0] == 0:  # TODO debug, why this case exists
            return np.zeros(shape=num_true_public_features), np.zeros(shape=num_true_public_features)

        faiss_res = faiss.StandardGpuResources()
        if mode == 'L2':
            index = faiss.IndexFlatL2(public_features.shape[1])
        # inner product; need normalization (https://github.com/spotify/annoy)
        elif mode == 'IP':
            index = faiss.IndexFlatIP(public_features.shape[1])
        elif mode == 'cos_sim':
            # normalize the embeddings first
            faiss.normalize_L2(public_features)
            faiss.normalize_L2(private_features)
            index = faiss.IndexFlatIP(public_features.shape[1])
        else:
            raise Exception(f'Unknown mode {mode}')
        if torch.cuda.is_available():
            index = faiss.index_cpu_to_gpu(faiss_res, 0, index)

        # logging.info(f'public_features shape : {public_features.shape}')
        # logging.info(f'private_features shape : {private_features.shape}')

        index.add(public_features.cpu().numpy())
        # logging.info(f'Number of samples in index: {index.ntotal}')

        distance, ids = index.search(private_features.cpu().numpy(), k=num_nearest_neighbor)
        # logging.info('Finished search')

        counter = Counter(list(ids.flatten()))
        # shape of the synthetic samples
        count = torch.zeros(num_true_public_features)
        for k in counter:
            count[k % num_true_public_features] += counter[k]
        # logging.info(f'Clean count: {count}')
        # logging.info(f'Clean count sum: {np.sum(count)}')
        # logging.info(f'Clean count num>0: {np.sum(count > 0)}')
        # logging.info(f'Largest clean counters: {sorted(count)[::-1][:50]}')
        #count = np.asarray(count)
        clean_count = count #.copy()
        # count += (np.random.normal(size=len(count)) * np.sqrt(num_nearest_neighbor) * noise_multiplier)
        # logging.info(f'Noisy count sum: {np.sum(count)}')
        # logging.info(f'Noisy count num>0: {np.sum(count > 0)}')
        # logging.info(f'Largest noisy counters: {sorted(count)[::-1][:50]}')
        # count = np.clip(count, a_min=threshold, a_max=None)
        # count = count - threshold
        # logging.info(f'Clipped noisy count sum: {np.sum(count)}')
        # logging.info(f'Clipped noisy count num>0: {np.sum(count > 0)}')
        # logging.info(f'Clipped largest noisy counters: {sorted(count)[::-1][:50]}')
        # torch.cuda.empty_cache()
        return clean_count  #, count



