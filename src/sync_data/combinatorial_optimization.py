import typing as tp
import torch
from src.sync_data.augmented_evolution import OffspringSelector
import torch.distributions as D
from src.sync_data.evaluators import PopulationWithEmbeddings
from src.sync_data.population import Population
from torch.optim import Adam
from scipy.special import digamma
import numpy as np
from tqdm import tqdm


class NonNormNNDistribution():
    def __init__(self, data_matrix, sigma=0.2):
        """ A mixture distribution with a weight differentiable sampling routine. """
        self.data_matrix = data_matrix
        self.sigma = sigma

    def log_prob(self, x: torch.tensor):
        org_dists = torch.cdist(x, self.data_matrix)
        #print(org_dists.shape)
        min_dists = torch.min(org_dists, dim=1)[0]
        return -0.5*(min_dists**2)/(self.sigma**2)

class WeightedDifferentiableMixture(D.MixtureSameFamily):
    def __init__(self, mixture_weights, component_distribution, validate_args=None, tsample=0.05):
        """ A mixture distribution with a weight differentiable sampling routine. """
        super().__init__(D.Categorical(mixture_weights.detach().clone()), component_distribution, validate_args)
        self.tsample = tsample
        self.mixture_weights = mixture_weights

    def log_prob(self, x):
        self._mixture_distribution = D.Categorical(logits=self.mixture_weights.detach().clone())
        return super().log_prob(x)

    def rsample(self, sample_shape=torch.Size()):
        """ Sample from all mixture components and do convex reweighting using a gumbel approximation. """
        comps = self._component_distribution.sample(sample_shape)
        if self.tsample == 0.0:
            weights = D.OneHotCategorical(logits=self.mixture_weights).sample(sample_shape)
        else:
            weights = D.RelaxedOneHotCategorical(self.tsample, logits=self.mixture_weights).rsample(sample_shape)
        return torch.sum(weights.unsqueeze(-1) * comps, axis=-2)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            self.rsample(sample_shape)


def empirical_kld(pdist, qdist, n_samples=200):
    """ A sampling based estimator of the KLD between two distributions. """
    mysamples = pdist.rsample(torch.Size([n_samples]))
    logpp = pdist.log_prob(mysamples)
    logpq = qdist.log_prob(mysamples)
    return torch.mean(logpp - logpq)


def mixture_from_population(pop: PopulationWithEmbeddings, radius=0.1, tsample=0.05, use_mixture_weights=None):
    dist_dict = {}
    for t in pop.tags:
        embs = pop.embeddings[t]
        components = D.Independent(D.Normal(embs, torch.ones_like(embs)*radius), 1)
        mixture_weights = torch.ones(len(embs))
        dist_dict[t] = WeightedDifferentiableMixture(mixture_weights, components, tsample=tsample)
    return dist_dict


def alpha_beta_from_correctness(corr, phi0):
    q = (2*corr-1)/(phi0-corr)
    #print(q)
    return q*phi0+1, q*(1-phi0)+1
def logor0(x):
    """ Log(x) if x > 0 else 0. """
    return np.log(x, out=np.zeros_like(x, dtype=np.float64), where=(x!=0))

def get_label_correctness_kld(p_agree, phi0, fallback = 5.0):
    """ Compute the expected label-correctness term for an individual sample according to the beta distribution model.
        p_agree should be a number between 0.5 and 1. phi0 is a number between 0 and 1 denoting initial label certainty.
    """
    npa = (p_agree < 1.0) & (phi0 != 0.5) #non_perfect_agreement

    ## Compute kld only for non perfect agreement, fill other field with 0.
    r1 = p_agree[npa]*phi0[npa]+(1-p_agree[npa])*(1-phi0[npa])
    alpha, beta = alpha_beta_from_correctness(r1, phi0[npa])
    h0 = -(logor0(phi0[npa]) * phi0[npa] + logor0(1-phi0[npa]) * (1-phi0[npa])) #if r0 > 0.0 else 0
    res = -phi0[npa] * digamma(alpha) - (1-phi0[npa]) * digamma(beta) + digamma(alpha + beta) - h0

    ret = np.zeros_like(p_agree)
    ret[npa] = res
    ret[phi0 == 0.5] = fallback
    return ret

class SGDOptimizationSelector(OffspringSelector):
    def __init__(self, target_population: PopulationWithEmbeddings, device="cuda", batch_size=1000,
                 temperature=0.05, target_radius=0.3, source_radius=0.2, utility_fn=None,
                 strategy="continous", sgd_steps=100, sgd_lr=3e-2, label_cert_weight=1.0) -> None:
        """ target_population: Population of unlabeled samples from the original domain.
            strategy: "continous" or "greedy".
        """
        self.device = device
        self.batch_size = batch_size
        self.target_population = target_population
        self.source_radius = source_radius
        self.target_mixtures = mixture_from_population(target_population, radius=target_radius, tsample=0.0)
        self.temperature = temperature
        self.utility_fn = utility_fn
        self.sgd_lr = sgd_lr
        self.sgd_steps = sgd_steps
        self.label_cert_weight = label_cert_weight

    def evaluate_objective(self, mypopulation_embedds: PopulationWithEmbeddings, n_sample_kld=1000):
        """ Only compute and return the loss in a dict per tag. """
        ret_dict = {}
        for tag in mypopulation_embedds.tags:
            embs = mypopulation_embedds.get_embeddings(tag)
            components = D.Independent(D.Normal(embs, torch.ones_like(embs) * self.source_radius), 1)
            mixture_weights = torch.ones(len(embs))  ## non normalized mixture weights.
            theta_distribution = WeightedDifferentiableMixture(mixture_weights, components, tsample=0.0) ## Discrete mixture distribution.
            ## Additional, per_sample penalties: KLD-Term + Utitity
            sample_penalties = self.label_cert_weight * torch.tensor(
                get_label_correctness_kld(mypopulation_embedds.get_agreement(tag),
                                          mypopulation_embedds.get_initial_prob(tag)), dtype=torch.float)
            if self.utility_fn is not None:
                sample_penalties += self.utility_fn(mypopulation_embedds[tag], embs)

            loss = empirical_kld(theta_distribution, self.target_mixtures[tag[0]], n_samples=n_sample_kld)
            loss += torch.sum(torch.softmax(mixture_weights, dim=-1) * sample_penalties)
            ret_dict[tag] = loss.item()
        return ret_dict

    def select(self, mypopulation_embedds: PopulationWithEmbeddings, num_select: int) -> tp.List[str]:
        ## Compute embeddings for Population
        ret_dict = {}
        #mypopulation_embedds = PopulationWithEmbeddings(population, embedding_model_str=self.embedding_model_str)
        for tag in mypopulation_embedds.tags:
            embs = mypopulation_embedds.get_embeddings(tag)
            components = D.Independent(D.Normal(embs, torch.ones_like(embs) * self.source_radius), 1)
            mixture_weights = torch.ones(len(embs)) ## non normalized mixture weights.

            mixture_weights.requires_grad_(True)

            theta_distribution = WeightedDifferentiableMixture(mixture_weights, components, tsample=self.temperature)
            ## Additional, per_sample penalties: KLD-Term + Utitity
            sample_penalties = self.label_cert_weight * torch.tensor(get_label_correctness_kld(mypopulation_embedds.get_agreement(tag),
                                                                      mypopulation_embedds.get_initial_prob(tag)), dtype=torch.float)
            if self.utility_fn is not None:
                sample_penalties += self.utility_fn(mypopulation_embedds, tag)

            myopt = Adam([mixture_weights], lr=self.sgd_lr)
            #n_remove = 500
            for steps in tqdm(range(self.sgd_steps)):
                myopt.zero_grad()
                loss = empirical_kld(theta_distribution, self.target_mixtures[tag[0]], n_samples=150)
                loss += torch.sum(torch.softmax(mixture_weights, dim=-1)*sample_penalties)

                # myp = torch.softmax(gmm_source.mixture_weights, dim=-1)
                # entropy_reg = torch.sum(myp[torch.argsort(myp)[:500]]) + torch.sum(torch.abs(myp[torch.argsort(-myp)[:100]]-0.01))
                # loss = loss + entropy_reg
                loss.backward()
                #if steps % 10 == 0:
                #    print(loss.detach())
                myopt.step()
            ret_dict[tag] = torch.softmax(mixture_weights.detach(), dim=-1).numpy()
        return ret_dict

class IndependentSelector():
    """ Solve an independent version of the optimization problem. """
    def __init__(self, target_population: PopulationWithEmbeddings, target_radius=0.3, source_radius=0.2,
                 utility_fn=None, label_cert_weight=1.0):
        self.target_population = target_population
        self.source_radius = source_radius
        self.target_radius = target_radius
        self.utility_fn = utility_fn
        self.label_cert_weight = label_cert_weight

    def evaluate_objective(self, mypopulation_embedds: PopulationWithEmbeddings):
        """ Only compute and return the loss in a dict per tag. """
        ret_dict = {}
        for tag in mypopulation_embedds.tags:
            loss = self._compute_sample_penalties(mypopulation_embedds, tag)
            ret_dict[tag] = torch.mean(loss).item()
        return ret_dict


    def _compute_sample_penalties(self, mypopulation_embedds: PopulationWithEmbeddings, tag):
        embs = mypopulation_embedds.get_embeddings(tag)
        ## Additional, per_sample penalties: KLD-Term + Utitity
        sample_penalties = self.label_cert_weight * torch.tensor(
            get_label_correctness_kld(mypopulation_embedds.get_agreement(tag),
                                      mypopulation_embedds.get_initial_prob(tag)), dtype=torch.float)
        #print("Label_correctness: ", sample_penalties.mean())
        if self.utility_fn is not None:
            sample_penalties += self.utility_fn(mypopulation_embedds, tag)
        ## Main scores.
        dist_mat = torch.cdist(embs, self.target_population.get_embeddings(tag[0]))
        min_dist, indices = torch.min(dist_mat, dim=1)  ## Closest target to each point.
        d = embs.shape[1]
        #print(d)
        const = 0.5 * d * np.log(2.0 * np.e * np.pi + self.source_radius * self.source_radius)
        sample_penalties += const + (min_dist * min_dist + d * self.source_radius * self.source_radius) / (
                    self.target_radius * self.target_radius)
        #print("Total: ", sample_penalties.mean())
        return sample_penalties

    def select(self, mypopulation_embedds: PopulationWithEmbeddings, num_select: int) -> tp.List[str]:
        ## Compute embeddings for Population
        ret_dict = {}
        # mypopulation_embedds = PopulationWithEmbeddings(population, embedding_model_str=self.embedding_model_str)
        for tag in tqdm(mypopulation_embedds.tags, desc='Selection'):
            sample_penalties = self._compute_sample_penalties(mypopulation_embedds, tag)
            ret_dict[tag] = torch.softmax(-sample_penalties, dim=-1).numpy()
        return ret_dict








