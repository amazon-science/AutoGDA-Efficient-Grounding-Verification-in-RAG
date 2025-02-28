import logging
import os
import csv
from typing import List, Tuple
from sentence_transformers import InputExample
import numpy as np
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, RocCurveDisplay, roc_curve, auc, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from src.utils.constants import COLOR, _BUCKET_NAME
from src.utils.model_s3_utils import upload_saved_model_to_s3
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# logger = logging.getLogger(__name__)
logger = logging.getLogger('my_logger')
print(__name__)


class CEBinaryClassificationEvaluatorWithBatching:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and binary labels (0 and 1),
    it compute the average precision and the best possible f1 score
    """

    def __init__(
        self,
        sentence_pairs: List[List[str]],
        labels: List[int],
        name: str = "",
        show_progress_bar: bool = False,
        save_path: str = "evaluation",
        write_csv: bool = True,
        save_best: bool = False,
        tokenizer = None
    ):
        assert len(sentence_pairs) == len(labels)

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name
        self.save_best = save_best
        self.best_loss = 0.0

        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.BCELoss()
        self.mse_loss_fn = nn.MSELoss()
        self.tokenizer = tokenizer

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        if write_csv:
            os.makedirs(save_path, exist_ok=True)
            self.csv_file = f"{save_path}/" + (name + "_" if name else "") + "results.csv"
            logger.info(f'Saving logs in {self.csv_file}')

        self.csv_headers = [
            "epoch",
            "steps",
            "Accuracy",
            "Accuracy_Threshold",
            "F1",
            "F1_Threshold",
            "Precision",
            "Recall",
            "Average_Precision",
            "AUC-ROC",
            "BCE Loss",
            "MSE Loss",
            "Pos Avg",
            "Neg Avg"
        ]
        self.write_csv = write_csv
        self.first_output = True




    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    @classmethod
    def from_input_tuples(cls, examples: List[Tuple], **kwargs):
        sentence_pairs = []
        labels = []

        for E, C, L in examples:
            sentence_pairs.append((E, C))
            labels.append(L)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, return_dict=False) -> float:

        logger.info("CEBinaryClassificationEvaluatorWithBatching: Evaluating the model on " + self.name)
        if isinstance(model, CrossEncoder):
            pred_scores = model.predict(
                self.sentence_pairs, convert_to_numpy=True, show_progress_bar=self.show_progress_bar, batch_size=1
            )
            S = torch.tensor(pred_scores, dtype=float)
        else: ## Automodelforsequenceclassification
            model.eval()
            device = list(model.parameters())[0].device
            with torch.no_grad():
                if type(model).__name__ == "HHEMv2ForSequenceClassification":
                    pair_dict = [{'text1': pair[0], 'text2': pair[1]} for pair in self.sentence_pairs]
                    x = self.tokenizer([model.prompt.format(**pair) for pair in pair_dict], return_tensors='pt', padding=True, truncation=True)
                else:
                    x = self.tokenizer(self.sentence_pairs, return_tensors='pt', padding=True, truncation=True)
                batch_sz = 4
                scores_list= []
                for b in tqdm(range(0, len(x["input_ids"]), batch_sz)):
                    output = model(input_ids=x["input_ids"][b:b+batch_sz].to(device), attention_mask=x["attention_mask"][b:b+batch_sz].to(device))
                    scores_list.append(torch.softmax(output.logits, dim=-1)[:, 1])
            S = torch.cat(scores_list).cpu()
            S = S.double()
            pred_scores = S.numpy()

        L = torch.tensor(self.labels, dtype=float)
        loss = float(self.loss_fn(S, L))
        mse = float(self.mse_loss_fn(S, L))
        degenerate = False
        if len(S[L.long() == 1]) == 0: # no positive labels
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
        # ------------ #
        # Compute metricsq
        # ------------ #
        acc, acc_threshold = BinaryClassificationEvaluator.find_best_acc_and_threshold(pred_scores, self.labels, True)

        f1, precision, recall, f1_threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(
            pred_scores, self.labels, True
        )
        if not degenerate:
            roc = roc_auc_score(self.labels, pred_scores)
        else:
            roc = -1.0
        ap = average_precision_score(self.labels, pred_scores)
        bacc = balanced_accuracy_score(self.labels, pred_scores > 0.5)
        logger.info("Accuracy:           {:.3f}\t(Threshold: {:.4f})".format(acc * 100, acc_threshold))
        logger.info("F1:                 {:.3f}\t(Threshold: {:.4f})".format(f1 * 100, f1_threshold))
        logger.info("Precision:          {:.3f}".format(precision * 100))
        logger.info("Recall:             {:.3f}".format(recall * 100))
        logger.info("Average Precision:  {:.3f}".format(ap * 100))
        logger.info("ROC-AUC:            {:.4f}".format(roc))
        logger.info("BCE Loss:           {:.4f}".format(loss))
        logger.info("MSE Loss:           {:.4f}".format(mse))
        logger.info("Positive Avg:       {:.4f}".format(pos_avg))
        logger.info("Negative Avg:       {:.4f}\n".format(neg_avg))
        ret = {"f1": float(f1), "accuracy": float(acc), "precision": float(precision),
               "recall": float(recall), "roc": float(roc), "pos_avg": float(pos_avg),
               "neg_avg": float(neg_avg), "bacc": float(bacc)}

        # ------------ #
        # Save best
        # ------------ #
        if self.save_best and roc > self.best_loss:
            logger.info(f"ROC is {roc:.5f}. Best prior ROC = {self.best_loss:.5f}.")
            logger.info(f"Saving model parameters in {output_path}")
            model.save(output_path)
            upload_saved_model_to_s3(bucket_name=_BUCKET_NAME, local_path=output_path, s3_path=output_path)
            self.best_loss = roc

        if output_path is not None and self.write_csv:
            if self.first_output:
                # ------------------ #
                # Remove old log file
                self.first_output = False
                output_file_exists = os.path.isfile(self.csv_file)
                if output_file_exists:
                    logger.info(f'Removing {self.csv_file}')
                    os.remove(self.csv_file)

            output_file_exists = os.path.isfile(self.csv_file)
            with open(self.csv_file, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                logger.info(f'Saving progress in {self.csv_file}')
                writer.writerow([epoch, steps, acc, acc_threshold, f1, f1_threshold, precision, recall, ap, roc, loss, mse, pos_avg, neg_avg])

        if return_dict:
            return ret
        else:
            return roc
