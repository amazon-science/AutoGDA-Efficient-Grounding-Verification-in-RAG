## Define evaluation metrics

from abc import ABC, abstractmethod
import typing as tp

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder
from sentence_transformers.evaluation import BinaryClassificationEvaluator

from src.sync_data.population import Population
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from src.utils.data_utils import AnnotatedTextDataset
from src.utils.data_utils import prepare_samples
from src.cross_encoder_model.my_evaluator import CEBinaryClassificationEvaluatorWithBatching
from src.cross_encoder_model.model_wrappers import TwoWayDebertaV2, TwoWayBart, DataCollatorWithTokenization, Vectara2DataCollatorWithTokenization, OneWayCrossEncoder, TwoWayT5
from transformers import Trainer, TrainingArguments
from scipy.special import softmax
from src.sync_data.compute_entailments import EntailmentCheckModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score
from vectara.modeling_vectara import HHEMv2ForSequenceClassification
import torch.nn as nn
from typing import Union
def add_summary_stats(metric_results):
    """ Add summary statistics to an array of metrics.
        :param: metric_results: Dict with key=metric_name, value=(dict with tags, metric results)
        Will add keys {metric_name}_avg and {metric_name}_std to the dict containing mean and standard deviation
        of the metric taken over the tags.
    """
    res_updates = {}
    for metric_name, tag_vals in metric_results.items():
        if isinstance(tag_vals, dict):
            values = np.array(list(tag_vals.values()))
            res_updates[f"{metric_name}_avg"] = values.mean()
            res_updates[f"{metric_name}_std"] = values.std()
        else:
            res_updates[f"{metric_name}_avg"] = metric_results[metric_name]
    metric_results.update(res_updates)
    return metric_results


def dict_tags_to_str(eval_single_res):
    """ Convert the dicts as tags to a single string for JSON compatibility. """
    for metric_name in eval_single_res.keys():
        if isinstance(eval_single_res[metric_name], dict):
            eval_single_res_new = {}
            for k, v in eval_single_res[metric_name].items():  # rewrite keys
                eval_single_res_new["_".join([str(kitem) for kitem in k])] = v
            eval_single_res[metric_name] = eval_single_res_new
    return eval_single_res


class PopulationWithEmbeddings(Population):
    """ A population which additionally stores the embeddings for each item.
        This class should be used to compute the evaluation metrics, without having to recompute sentence embeddings.
    """

    def __init__(self, population: Population, embedding_model_str="sentence-t5-base", device="cuda", init_embeddings=True):
        """ Compute the embeddings corresponding to a population.
            This can be used to compute the metrics.
        """
        self.target_population = population.target_population
        self.model_str = embedding_model_str
        self.population = population
        self.batch_size = 64
        self.device = device

        ## compute embeddings
        self.embeddings = {}
        if init_embeddings:
            self.model = SentenceTransformer(self.model_str, device=self.device)  # ='cuda',
            self.model.eval()
            for tag in tqdm(population.tags):
                self.embeddings[tag] = self._extract_features(population[tag])
            self.model = None
        self.agreement_frac = population.agreement_frac
        self.references = population.references
        self.init_dist = population.init_dist


    def remove_duplicates(self):
        print("Depricated, not implemented for embeddings.")
        super().remove_duplicates()

    def get_embeddings(self, tag):
        """ Return matrix of embeddings corresponding to the samples in the population with a
            specific tag.
        """
        if tag in self.embeddings:
            return self.embeddings[tag]
        else:
            return None

    def _extract_features(self, data):
        # If available, the model is automatically executed on the GPU.
        batch_size = self.batch_size
        with torch.no_grad():
            sentence_embeddings = []
            for i in range(len(data) // batch_size + 1):
                embeddings = self.model.encode(
                    data[i * batch_size:(i + 1) * batch_size])
                if len(embeddings) > 0:
                    sentence_embeddings.append(embeddings)
        sentence_embeddings = np.concatenate(sentence_embeddings)
        return torch.from_numpy(sentence_embeddings).float()

    def __add__(self, other: "PopulationWithEmbeddings"):
        super().__add__(other)
        for t in self.tags:
            if other.get_embeddings(t) is not None:
                if self.get_embeddings(t) is not None:
                    self.embeddings[t] = torch.cat((
                        self.embeddings[t], other.get_embeddings(t)), dim=0)
                else:
                    self.embeddings[t] = other.get_embeddings(t)
        return self

    def get_indexed_subpopulation(self, index_dict):
        subpopulation = PopulationWithEmbeddings(super().get_indexed_subpopulation(index_dict), init_embeddings=False)
        for tag in self.get_tags():
            if tag in index_dict and self.get_embeddings(tag) is not None:
                subpopulation.embeddings[tag] = self.get_embeddings(tag)[torch.tensor(index_dict[tag])]
        return subpopulation

class Evaluator(ABC):
    """ Implement evaluation metrics to compute the distance between source and target population. """
    @abstractmethod
    def __call__(self, source_population: PopulationWithEmbeddings,
                 target_population: PopulationWithEmbeddings) -> tp.Dict[str, float]:
        raise NotImplementedError("Override this method in the Evaluator implementation.")


class NNCoverageEvaluator(Evaluator):
    """ Evaluate the coverage of the distribution, by computing the share
        that are the nearest neighbor to at least one target sample.
        The score should be high, if the source samples are scattered all over the population.
        If only a small number of samples are near to the population, then only them will be a nearest neighbor to
        the target samples, and the others will not be a nearest neighbor, results in low NNCoverage.
        There are two scores:
    """
    def __init__(self, normalize=False):
        """:param: normalize: normalize the embeddings before the computations. """
        self.normalize = normalize

    def __call__(self, source_population, target_population):
        unique_metric_list = {}
        min_distance_list = {}
        for t in target_population.tags:
            target_embeddings = target_population.get_embeddings(t)
            population_embeddings = source_population.get_embeddings(t)
            if self.normalize:
                target_embeddings = target_embeddings/torch.norm(target_embeddings, dim=1, keepdim=True)
                population_embeddings = target_embeddings/torch.norm(population_embeddings, dim=1, keepdim=True)
            dist_mat = torch.cdist(target_embeddings, population_embeddings)
            min_dist, indices = torch.min(dist_mat, dim=1) ## Closest one to each target point.
            uniques = torch.unique(indices)
            unique_metric_list[t] = len(uniques)/len(population_embeddings)
            min_distance_list[t] =torch.mean(min_dist).item()

        return {"min_distance": min_distance_list,
                "unique_nns": unique_metric_list}



def matrix_eig_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the square root of the eigenvalues. Take real part for potential negative eigenvalues.
    """
    vals = torch.linalg.eigvals(matrix)
    #vals = torch.view_as_complex(vals.contiguous()
    return vals.sqrt().real

class FrechetInceptionDistance():
    """ Compute FID. Adapted from https://github.com/AI-secure/aug-pe/blob/main/utility_eval/precision_recall.py"""

    def __call__(self, source_population, target_population):
        ret_dict = {}
        for t in target_population.tags:
            # calculate mean and covariance statistics
            mu1, sigma1 = source_population.get_embeddings(t).mean(axis=0), source_population.get_embeddings(t).t().cov()
            mu2, sigma2 = source_population.get_embeddings(t).mean(axis=0), source_population.get_embeddings(t).t().cov()
            # calculate sum squared difference between means
            ssdiff = torch.sum((mu1 - mu2).pow(2.0))
            # calculate sqrt of product between cov

            eigsum = matrix_eig_sqrt(sigma1.matmul(sigma2)).sum() #, 0.5)
            # calculate score
            fid = ssdiff + torch.trace(sigma1) + torch.trace(sigma2) - 2.0 * eigsum
            ret_dict[t] = fid.item()
        return {"fid": ret_dict}

class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self, features, nhood_sizes=[3], clamp_to_percentile=None, eps=1e-5):
        """Estimate the manifold of given feature vectors.

            Args:
                distance_block: DistanceBlock object that distributes pairwise distance
                    calculation to multiple GPUs.
                features (torch.tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self._ref_features = features

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = torch.cdist(features, features)
        self.D = torch.topk(self.D, k=max(self.nhood_sizes)+1,
                            largest=False).values[:, torch.tensor(self.nhood_sizes)] # contains the distance

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        batch_predictions = torch.cdist(eval_features, self._ref_features)  # closest distances (n_eval, n_ref)
        samples_in_manifold = batch_predictions <= self.D.reshape(1, -1) # Are they closer than the min distanc required?

        #if return_realism and return_neighbors:
        #    return batch_predictions, max_realism_score, nearest_indices
        #elif return_realism:
        #    return batch_predictions, max_realism_score
        #elif return_neighbors:
        #    return batch_predictions, nearest_indices

        return samples_in_manifold.any(dim=1)

class ManifoldClassification():
    """ Metrics relying on classification of whether a synthetic sample is in manifold for
        the target samples. Apdapted from https://github.com/AI-secure/aug-pe/blob/main/utility_eval/precision_recall.py
        We compute precision, recall and f1-scores for this classification problem.
    """
    def __init__(self, nhoods: tp.List[int] = [3]):
        self.nhoods = nhoods

    def __call__(self, source_population, target_population):
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        for t in target_population.tags:
            src_manifold = ManifoldEstimator(source_population.get_embeddings(t), self.nhoods)
            trg_manifold = ManifoldEstimator(target_population.get_embeddings(t), self.nhoods)

            state = []
            # Precision: How many points from eval_features are in ref_features manifold.
            precision = trg_manifold.evaluate(source_population.get_embeddings(t))
            precision_dict[t] = precision.float().mean(axis=0).item()

            # Recall: How many points from ref_features are in eval_features manifold.
            recall = src_manifold.evaluate(target_population.get_embeddings(t))
            recall_dict[t] = recall.float().mean(axis=0).item()
            if precision_dict[t] + recall_dict[t] > 0:
                f1_dict[t] = (2 * precision_dict[t] * recall_dict[t]) / \
                              (precision_dict[t] + recall_dict[t])
            else:
                f1_dict[t] = 0.0

        return {"precision": precision_dict, "recall": recall_dict, "f1": f1_dict}


class PerplexityEvaluation():
    """ Compute the perplexity of the output via some MaskedLM or Seq2Seq """
    def __init__(self,  model_str="FacebookAI/roberta-large", tokenizer_str="FacebookAI/roberta-large", device="cuda"):
        self.device = device
        # t5_style indicates whether a MaskedLM or Seq2Seq Architecture is used.
        self.t5_style = ("t5" in model_str)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        if not self.t5_style:  # Masked LM
            self.model = AutoModelForMaskedLM.from_pretrained(model_str).to(device)
        else:  # Seq2Seq LM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_str).to(device)
        self.batch_size = 64
    def _prepare_inputs(self, sentences):
        """ prepare input tokens, attention_masks, and labels for forward pass to compute perplexity. """
        if not self.t5_style:
            mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        else:
            mask_token = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        tensor_input = self.tokenizer(sentences, return_tensors='pt', padding=True)
        valid_mask = tensor_input["attention_mask"].clone()
        repeat_input = tensor_input["input_ids"].unsqueeze(1).repeat(1, tensor_input["input_ids"].size(-1), 1)
        repeat_mask = tensor_input["attention_mask"].unsqueeze(1).repeat(1, tensor_input["attention_mask"].size(-1), 1)
        mask = torch.ones(repeat_input.size(-1)).diag(0).unsqueeze(0)
        ## Invalidate last token
        seq_lens = torch.sum(valid_mask, dim=1)
        valid_mask[torch.arange(len(seq_lens)), seq_lens - 1] = 0
        if not self.t5_style:
            ## First token should not be considered for perplexity as well for BERT/ROBERTA
            labels = repeat_input.clone().masked_fill(mask == 0, -100)
            valid_mask[:, 0] = 0
        else:
            ## We do not need to consider every row to compute perplexity
            valid_mask[torch.arange(len(seq_lens)), seq_lens - 1] = 0
            labels = mask_token * torch.ones(repeat_input.size(0), repeat_input.size(1), 2, dtype=torch.long)
            labels[:, :, 1] = tensor_input["input_ids"]
        masked_input = repeat_input.masked_fill(mask == 1, mask_token)
        return masked_input, repeat_mask, labels, valid_mask

    def _compute_pplx(self, masked_input, repeat_mask, labels, valid_mask):
        """ Compute perplexity over valid tokens. """
        masked_input = masked_input.reshape(-1, masked_input.size(-1))
        repeat_mask = repeat_mask.reshape(-1, repeat_mask.size(-1))
        labels = labels.reshape(-1, labels.size(-1))
        masked_input = masked_input[valid_mask.flatten() == 1]
        repeat_mask = repeat_mask[valid_mask.flatten() == 1]
        labels = labels[valid_mask.flatten() == 1]
        with torch.no_grad():
            tot_loss = 0.0
            for b in tqdm(range(0, len(masked_input), self.batch_size)):
                ret = self.model(input_ids=masked_input.to(self.device)[b:b+self.batch_size],
                                         attention_mask=repeat_mask.to(self.device)[b:b+self.batch_size],
                                         labels=labels.to(self.device)[b:b+self.batch_size])
                tot_loss += ret.loss.detach().cpu().item()*len(masked_input[b:b+self.batch_size])
        return tot_loss/len(masked_input)

    def __call__(self, source_population, target_population):
        pplx_dict = {}
        for t in source_population.tags:
            masked_input, repeat_mask, labels, valid_mask = self._prepare_inputs(source_population[t].tolist())
            pplx_dict[t] = self._compute_pplx(masked_input, repeat_mask, labels, valid_mask)

        return {"perplexity": pplx_dict}


def dataset_from_population(source_population: Population):
    """
        Convert Population into Annotated Text Dataset.
    """
    df_source = source_population.to_dataframe()
    df_source["evidence"] = df_source["tag_0"]
    df_source["label_binary"] = df_source["tag_1"]
    df_source["claim"] = df_source["sample"]
    df_source['id'] = df_source.index
    df_source["label"] = df_source["label_binary"]
    df_source["query"] = "paraphrase"
    df_source["group"] = "dummy"
    if "p_init" in df_source:
        df_source["label_cert"] = df_source["p_init"]
    if "p_agree" in df_source:
        df_source["label_cert"] = df_source["label_cert"]*df_source["p_agree"] + (1-df_source["label_cert"])*(1-df_source["p_agree"])
    dataset = AnnotatedTextDataset(df_source, data_id="current_synth")
    return dataset

class TrainerWithWeightedCELoss(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        del inputs["labels"]
        if "weight" in inputs:
            weights = inputs["weight"]
            del inputs["weight"]
        outputs = model(**inputs)
        if self.weighted:
            loss = torch.mean(torch.nn.CrossEntropyLoss(reduction='none').forward(outputs.logits, labels)*weights)
        else:
            loss = torch.mean(torch.nn.CrossEntropyLoss(reduction='none').forward(outputs.logits, labels))
        return (loss, outputs) if return_outputs else loss

    def set_weighted(self, weighted):
        self.weighted = weighted

class NLIFinetuningEvaluation():

    """ Finetune an NLI model on the synthetic samples and measure its performance."""
    def __init__(self, test_dataset, target_model_str: Union[str, EntailmentCheckModel] = "tasksource",
                 num_labels=1, device="cuda", eval_target=False, num_epochs=2, is_entailment_check=False,
                 query_mode = None, batch_size=1, lr=1e-5, renew_model=True, weighted=False):
        """
            :param: target_model_str: The model to finetune in this evaluation.
                Current models that support finetuning are: vectara_v1, vectara_v2, bart-large, nli-deberta-v3-base, tasksource, tasksource_v2.

            :param: eval_target:
            :param: is_entailment_check, if True use implementations from EntailmentCheckModel, those cannot be finetuned
            :param: query mode: How to integrate the queries from the original dataset. For the test dataset, evidence must already
                be in the same format as the queries.
        """
        self.target_model_str = target_model_str
        self.is_entailment_check = is_entailment_check
        self.device = device
        self.eval_target = eval_target
        self.test_dataset = test_dataset
        ## Short names to long names map.
        if not is_entailment_check: ## Prepare to do actual finetuning
            self.models_map = {
                "nli-deberta-v3-base": "cross-encoder/nli-deberta-v3-base",
                "vectara_v1": "vectara/hallucination_evaluation_model",
                "vectara_v2": "vectara/hallucination_evaluation_model",
                "tasksource_v1": "tasksource/deberta-base-long-nli",
                "tasksource": "tasksource/deberta-base-long-nli",
                "bart-large": "facebook/bart-large-mnli",
                "flan-t5-base": "sjrhuschlee/flan-t5-base-mnli"
            }
            self.num_epochs = num_epochs
            self.num_labels = num_labels
            self.batch_size = batch_size
            self.learning_rate = lr
            if self.target_model_str == "flan-t5-base":
                self.learning_rate = 20*lr ## Adapt for t5 which required substantially higher lr
            self.crossencoder = self.target_model_str in ["nli-deberta-v3-base", "vectara_v1"]
        self.renew_model = renew_model
        self.weighted = weighted
        self.model, self.tokenizer = None, None

    def _prepare_model(self):
        """ Load the model that will be finetuned. """
        if isinstance(self.target_model_str, EntailmentCheckModel):
            return self.target_model_str.model, self.target_model_str.tokenizer
        if self.is_entailment_check:
            if self.target_model_str.endswith("-split"):
                print("Splitting")
                model = EntailmentCheckModel(self.target_model_str[:-6], device=self.device, split_sentences=True)
            else:
                model = EntailmentCheckModel(self.target_model_str, device=self.device)
            return model, None
        else:
            if self.crossencoder:
                if self.target_model_str == "vectara_v1":
                    revision='hhem-1.0-open'
                    model = CrossEncoder(
                        self.models_map[self.target_model_str],
                        num_labels=self.num_labels,
                        automodel_args={"ignore_mismatched_sizes": True},
                        device=self.device,
                        trust_remote_code = True,
                        revision = revision
                    )
                elif self.target_model_str == "nli-deberta-v3-base":
                    model = OneWayCrossEncoder(
                        'cross-encoder/nli-deberta-v3-base',
                        num_labels=3,
                        device=self.device,
                        automodel_args={"ignore_mismatched_sizes": True},
                    )
                tokenizer = None
            elif self.target_model_str == "vectara_v2":
                # if train_layers is not None:
                model = HHEMv2ForSequenceClassification.from_pretrained(
                    'vectara/hallucination_evaluation_model', trust_remote_code=True)
                model = model.to(self.device)
                tokenizer = model.tokenzier #sic
                return model, tokenizer
            else: ## Huggingface models
                if self.target_model_str == "tasksource" or self.target_model_str == "tasksource_v1":
                    revision=None
                    if self.target_model_str == "tasksource_v1":
                        revision="9f72a81a6be78ec99f78314c4636b3a60259ab30"
                    model = TwoWayDebertaV2.from_pretrained(self.models_map[self.target_model_str], revision=revision)
                    tokenizer = AutoTokenizer.from_pretrained(self.models_map[self.target_model_str], revision=revision)
                elif self.target_model_str == "bart-large":
                    model = TwoWayBart.from_pretrained(self.models_map[self.target_model_str])
                    tokenizer = AutoTokenizer.from_pretrained(self.models_map[self.target_model_str])
                elif self.target_model_str == "flan-t5-base":
                    model = TwoWayT5.from_pretrained(self.models_map[self.target_model_str])
                    tokenizer = AutoTokenizer.from_pretrained(self.models_map[self.target_model_str])
                else:
                    raise ValueError(f"Unknown model {self.target_model_str}")
                #elif self.target_model_str == "vectara_v2":
                #    model = AutoModelForSequenceClassification.from_pretrained(
                #        'vectara/hallucination_evaluation_model', trust_remote_code=True)
                #    tokenizer = model.tokenzier
                model = model.to(self.device)
            return model, tokenizer


    def _eval_entailment_model(self, source_population):
        """ Run evaluation using an entailment checking model. NO FINETUNING IS DONE. """
        # dataset = dataset_from_population(source_population)
        pairs = list(zip(self.test_dataset.df.evidence, self.test_dataset.df.claim))
        scores = self.model.compute_scores(pairs)
        labels = self.test_dataset.df.label_binary.values

        loss_fn = nn.BCELoss()
        mse_loss_fn = nn.MSELoss()
        S = torch.tensor(scores, dtype=torch.float)
        # actual eval
        L = torch.tensor(labels, dtype=torch.long)
        degenerate = False
        if len(S[L.long() == 1]) == 0:  # no positive labels
            print("Degenerate evaluation. No positive labels in dataset!")
            degenerate = True
            pos_avg = -1.0
        else:
            pos_avg = torch.mean(S[L.long() == 1]).item()
        if len(S[L.long() == 0]) == 0:
            print("Degenerate evaluation. No negative labels in dataset!")
            degenerate = True
            neg_avg = -1.0
        else:
            neg_avg = torch.mean(S[L.long() == 0]).item()

        acc, acc_threshold = BinaryClassificationEvaluator.find_best_acc_and_threshold(scores, labels, True)

        f1, precision, recall, f1_threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(
            scores, labels, True
        )
        if not degenerate:
            roc = roc_auc_score(labels, scores)
        else:
            roc = -1.0
        ap = average_precision_score(labels, scores)
        bacc = balanced_accuracy_score(labels, scores > 0.5)
        ret = {"f1": float(f1), "accuracy": float(acc), "precision": float(precision),
               "recall": float(recall), "roc": float(roc), "pos_avg": float(pos_avg),
               "neg_avg": float(neg_avg), "bacc": float(bacc)}
        return ret

    def refit_and_eval(self, source_population, num_epochs=1):
        dataset = dataset_from_population(source_population)
        train_samples, val_samples, test_samples = prepare_samples(dataset.df, self.test_dataset.df,
                                                                   label_col='label_binary', do_val=False)

        N_insample_evaluation_size = int(len(train_samples) * 0.1)
        train_eval_samples = train_samples  # [N_insample_evaluation_size]
        # train_evaluator = CEBinaryClassificationEvaluatorWithBatching.from_input_examples(train_eval_samples,
        #                                                                                 name=f"TRAIN",
        #                                                                                  write_csv=False,
        #                                                                                  save_best=False)
        test_evaluator = CEBinaryClassificationEvaluatorWithBatching.from_input_examples(test_samples,
                                                                                         name=f"TRAIN",
                                                                                         write_csv=False,
                                                                                         save_best=False,
                                                                                         tokenizer=self.tokenizer)
        train_dataloader = DataLoader(train_samples, self.batch_size, shuffle=True)

        if self.crossencoder:
            self.model.fit(
                train_dataloader=train_dataloader,
                evaluator=test_evaluator,
                evaluation_steps=-1,
                warmup_steps=100,
                epochs=num_epochs,
                output_path=None,
                show_progress_bar=True,
                save_best_model=True,
                optimizer_params={"lr": self.learning_rate},
            )
        else:
            if self.target_model_str == "vectara_v2":
                data_collator = Vectara2DataCollatorWithTokenization(tok=self.tokenizer, prompt=self.model.config.prompt)
            else:
             # "Common" AutoModels
                data_collator = DataCollatorWithTokenization(tok=self.tokenizer)

            training_args = TrainingArguments(output_dir=".", learning_rate=self.learning_rate, eval_steps=-1,
                                              num_train_epochs=num_epochs, eval_strategy="no", save_steps=-1,
                                              remove_unused_columns=False, label_names=["labels"],
                                              per_device_train_batch_size=self.batch_size)
            import evaluate
            bin_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
            score_metrics = evaluate.combine(["roc_auc"])

            def huggingface_eval(eval_pred):
                if isinstance(eval_pred.predictions, tuple):
                    preds_scores = eval_pred.predictions[0]
                else:
                    preds_scores = eval_pred.predictions
                preds_scores = softmax(preds_scores, axis=-1)
                metrics_dict = bin_metrics.compute(predictions=preds_scores[:, 1] > 0.5, references=eval_pred.label_ids)
                metrics_dict.update(
                    score_metrics.compute(prediction_scores=preds_scores[:, 1], references=eval_pred.label_ids))
                return metrics_dict
            print("init trainer.")

            if self.target_model_str == "vectara_v2":
                model_train = self.model.t5
            else:
                model_train = self.model
            print(type(model_train))
            torch.cuda.synchronize()
            trainer = TrainerWithWeightedCELoss(
                model=model_train,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=self.test_dataset,
                compute_metrics=huggingface_eval,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            trainer.set_weighted(self.weighted)
            trainer.train()

        return test_evaluator(self.model, output_path=None, return_dict=True)

    def __call__(self, source_population, target_population):
        ## get a fresh model
        if self.model is None or self.renew_model:
            self.model, self.tokenizer = self._prepare_model()
        if self.is_entailment_check:
            res = self._eval_entailment_model(source_population)
        else:
            res = self.refit_and_eval(source_population, self.num_epochs)
        if self.renew_model:
            self.model, self.tokenizer = None, None ## Free memory by deleting model
        return res


class VectaraInferenceEvaluation():
    """ Evaluate inference scores using the original evidence and a NLI model loaded from EC2. """
    def __init__(self, target_model_local_path, num_labels=1, device="cuda", eval_target=False):
        """
            :param: eval_target:
        """

        self.model = CrossEncoder(target_model_local_path, num_labels=num_labels, device=device)
        self.eval_target = eval_target

    def __call__(self, source_population, target_population):
        vect_dict = {}
        target_dict = {}
        for t in source_population.tags:
            sentence_pairs = [[t[0], sample] for sample in source_population[t]]
            res = self.model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False, batch_size=4)
            vect_dict[t] = float(res.mean())
            if self.eval_target:
                sentence_pairs = [[t[0], sample] for sample in target_population[t]]
                res = self.model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False, batch_size=4)
                target_dict[t] = float(res.mean())

        if self.eval_target:
            return {"nli_score": vect_dict, "nli_score_target": target_dict}
        else:
            return {"nli_score": vect_dict}