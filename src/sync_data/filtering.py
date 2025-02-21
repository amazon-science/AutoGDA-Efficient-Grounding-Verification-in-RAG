## Filtering step to preserve the labels.
from sentence_transformers import CrossEncoder
from src.sync_data.population import Population
from src.utils.data_utils import AnnotatedTextDataset
from src.cross_encoder_model.fine_tuning_crossencoder import prepare_samples
from torch.utils.data import DataLoader
import numpy as np

class NLIModelEntailmentCheck():
    """ Use and NLI model to check if the mutated samples preserve entailment. """

    def __init__(self, target_model_local_path: str, threshold = 0.9, filter_higher = True, num_labels=1,
                 device="cuda", check_against_claim=False, min_keep=5):
        """ Initialize new NLI model from a checkpoint. """
        self.model = CrossEncoder(target_model_local_path, num_labels=num_labels, device=device)
        self.check_against_claim = check_against_claim
        self.filter_higher = filter_higher
        self.threshold = threshold
        self.batch_size = 2
        self.min_keep = min_keep

    def finetune_nli(self, train_population: Population, epochs=1, lr=1e-5):
        """ Update the NLI model with the current population. """
        df_source = train_population.to_dataframe()
        df_source["evidence"] = df_source["tag_0"]
        df_source["label_binary"] = df_source["tag_1"]
        df_source["claim"] = df_source["sample"]
        df_source['id'] = df_source.index
        df_source["label"] = df_source["label_binary"]
        df_source["query"] = "paraphrase"
        df_source["group"] = "dummy"

        dataset = AnnotatedTextDataset(df_source, data_id="current_synth")
        train_samples, val_samples, test_samples = prepare_samples(dataset.df, self.test_dataset.df,
                                                                   label_col='label_binary', do_val=False)
        ## get a fresh model
        train_dataloader = DataLoader(train_samples, self.batch_size, shuffle=True)

        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=None,
            evaluation_steps=len(train_dataloader),
            warmup_steps=100,
            epochs=epochs,
            output_path=None,
            show_progress_bar=True,
            save_best_model=False,
            optimizer_params={"lr": lr}
        )

    def filter_entailment_scores(self, current_population: Population, parent_population: Population):
        for t in current_population.tags:
            if self.check_against_claim:
                sentence_pairs = [[t[0], sample] for sample in current_population[t]]
                res = self.model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False, batch_size=4)
            else:
                sentence_pairs = [[parent_population[t][ref], sample] for sample, ref in
                                  zip(current_population[t], current_population.get_references(t))]
                res = self.model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False, batch_size=4)
            #print(res.mean())
            if self.filter_higher:
                scores = np.sort(res, axis=0)[::-1]
                use_threshold = min(self.threshold, scores[min(self.min_keep, len(scores))-1])
                current_population[t] = (current_population[t][res >= use_threshold],
                                         current_population.get_references(t)[res >= use_threshold])
            else:
                scores = np.sort(res, axis=0)
                use_threshold = max(self.threshold, scores[min(self.min_keep, len(scores))-1])
                current_population[t] = (current_population[t][res <= use_threshold],
                                         current_population.get_references(t)[res <= use_threshold])
        return current_population