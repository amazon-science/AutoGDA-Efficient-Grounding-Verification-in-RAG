import torch
import sys
import logging
import os
import csv
import argparse
from typing import List
import numpy as np
import pandas as pd
import evaluate
import torch.nn as nn

#sys.path.append("../")
from src.utils.global_variables import COLOR, _BUCKET_NAME
from src.utils.data_utils import AnnotatedTextDataset, DATA_LOADER_FN
from src.cross_encoder_model.model_wrappers import TwoWayDebertaV2, TwoWayBart, DataCollatorWithTokenization
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import average_precision_score
from sentence_transformers import InputExample, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from src.cross_encoder_model.my_evaluator import CEBinaryClassificationEvaluatorWithBatching
from src.utils.model_s3_utils import upload_saved_model_to_s3
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import evaluate
from scipy.special import softmax


from sentence_transformers.cross_encoder.evaluation import (
    CEF1Evaluator,
    CESoftmaxAccuracyEvaluator,
    CEBinaryClassificationEvaluator,
)
from sentence_transformers.evaluation import (
    SequentialEvaluator,
    SentenceEvaluator,
    BinaryClassificationEvaluator,
)


#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[LoggingHandler()],
)
#logger = logging.getLogger(__name__)
logger = None


def filter_long_token_sizes(examples: List, model_name: str ):
    new_examples = []
    sentence_pairs = []
    labels = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    Sizes = []
    ch_size = []
    for example in examples:
        input = example.texts[0] + example.texts[1]
        output = tokenizer(input, return_tensors="np")
        size = len(output['input_ids'][0])
        Sizes.append(size)
        ch_size.append(len(input))
        if size < 512:
            new_examples.append(example)
    return new_examples


def freeze_layers(model, train_layers: List[str]):
    """

    -05-21 19:54:02 - Freezed layer: deberta.encoder.layer.11.output.LayerNorm.bias
    2024-05-21 19:54:02 - Freezed layer: deberta.encoder.rel_embeddings.weight
    2024-05-21 19:54:02 - Freezed layer: deberta.encoder.LayerNorm.weight
    2024-05-21 19:54:02 - Freezed layer: deberta.encoder.LayerNorm.bias
    2024-05-21 19:54:02 - Freezed layer: pooler.dense.weight
    2024-05-21 19:54:02 - Freezed layer: pooler.dense.bias
    2024-05-21 19:54:02 - Training layer: classifier.weight
    2024-05-21 19:54:02 - Training layer: classifier.bias

    :param model:
    :param train_layers:
    :return:
    """
    train_params= 0
    for name, param in model.model.named_parameters():
        training_layer = False
        if train_layers is None:
            training_layer = True
        else:
            for  layer_prefix in train_layers:
                if name.startswith(layer_prefix):
                    training_layer = True
                    break

        if training_layer:  # choose whatever you like here
            train_params += np.prod(param.size())
            logger.info(f'Training layer: {name}')
            param.requires_grad = True
        else:
            # logger.info(f'Freezed layer: {name}')
            param.requires_grad = False
    print(f'Train params = {train_params}')

def prepare_samples(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str, do_val=True):

    train_samples, val_samples, test_samples = [], [], []

    train_df = train_df.sample(frac=1, random_state=0)
    if do_val:
        N_train = int(len(train_df) * 0.8)
    else:
        N_train = len(train_df) ## all for train

    train2_df = train_df.iloc[:N_train, :]
    val_df = train_df.iloc[N_train:, :]

    # Use 80% of the train data for training
    for e, c, l in zip(train2_df["evidence"], train2_df["claim"], train2_df[label_col]):
        train_samples.append(InputExample(texts=[e, c], label=int(l)))

    # Hold 20% of the train data for validation
    for e, c, l in zip(val_df["evidence"], val_df["claim"], val_df[label_col]):
        val_samples.append(InputExample(texts=[e, c], label=int(l)))

    # TEST DATASET
    for e, c, l in zip(test_df["evidence"], test_df["claim"], test_df[label_col]):
        test_samples.append(InputExample(texts=[e, c], label=int(l)))

    return train_samples, val_samples, test_samples


def fine_tune_train(
        dataset: AnnotatedTextDataset,
        test_dataset: List[AnnotatedTextDataset],
        base_model_name: str,
        save_model_name: str,
        num_epochs: int,
        learning_rate: float,
        label_col: str,
        per_device_train_batch_size: int = None,
        cuda_device_id: int = None,
        train_layers: List[str] = None,
        save_params: bool = True,
        crossencoder=True,
        token_limit = None,
        device="cuda"
    ):

    train_dataset_name = dataset.get_dataset_identifier()
    #test_dataset_name = test_dataset

    print(f'Using label-col={label_col}..')
    # TRAINING DATASET
    nli_problem = label_col == 'label'

    print(f'train dataset', train_dataset_name)
    test_samples_list = []
    for ts in test_dataset:
        train_samples, val_samples, test_samples = prepare_samples(dataset.df, ts.df, label_col, do_val = False)
        test_samples_list.append(test_samples)
    ## Only evaluate on small examples
    # dev_samples = filter_long_token_sizes(dev_samples, base_model_name)

    print("Num train samples: ", len(train_samples))
    print("Num val samples: ", len(val_samples))
    print("Num test samples: ", list([len(t) for t in test_samples_list]))


    if crossencoder:
        train_dataloader = DataLoader(train_samples, per_device_train_batch_size, shuffle=True)
        model = CrossEncoder(
            base_model_name,
            num_labels=1 if not nli_problem else len(set(dataset.df[label_col])),
            automodel_args={"ignore_mismatched_sizes": True},
            device=torch.device("cuda", cuda_device_id) if cuda_device_id is not None else None,
        )

        # if train_layers is not None:
        freeze_layers(model, train_layers)
        tokenizer = None
    else:
        train_dataloader = DataLoader(dataset, per_device_train_batch_size, shuffle=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if base_model_name == "tasksource/deberta-base-long-nli":
            model = TwoWayDebertaV2.from_pretrained(base_model_name)
        elif base_model_name == "facebook/bart-large-mnli":
            model = TwoWayBart.from_pretrained(base_model_name)
        model.to(device)

    # Number of steps per epoch on the train_dataloader
    num_steps_per_epoch = len(train_dataloader.dataset) // train_dataloader.batch_size + int(
        len(train_dataloader.dataset) % train_dataloader.batch_size != 0
    )
    print("Number of steps per epoch:", num_steps_per_epoch)

    N_insample_evaluation_size = max(int(len(train_samples) * 0.1), 300)
    train_eval_samples = train_samples[:N_insample_evaluation_size]  # Use %10 of train samples to track train error
    train_evaluator = CEBinaryClassificationEvaluatorWithBatching.from_input_examples(train_eval_samples,
                                                                                      name=f"TRAIN",
                                                                                      save_path=f"evaluation/{save_model_name}",
                                                                                      write_csv=True,
                                                                                      save_best=False,
                                                                                      tokenizer=tokenizer)
    val_evaluator = CEBinaryClassificationEvaluatorWithBatching.from_input_examples(val_samples, name=f"VAL",
                                                                                        save_path=f"evaluation/{save_model_name}",
                                                                                        write_csv=True,
                                                                                        save_best=save_params,
                                                                                        tokenizer=tokenizer)
    test_evaluators = []
    for ts, tset in zip(test_samples_list, test_dataset):
        test_evaluators.append(CEBinaryClassificationEvaluatorWithBatching.from_input_examples(ts, name=f"TEST_{tset.get_dataset_identifier().replace('/', '-')}",
                                                                                        save_path=f"evaluation/{save_model_name}",
                                                                                        write_csv=True,
                                                                                        save_best=False,
                                                                                        tokenizer=tokenizer))

    test_evaluator = SequentialEvaluator(test_evaluators)
    print(f'============================')
    print(f'==== Initial evaluation ====')
    test_evaluator(model, output_path=".", epoch=-1)
    print(f'============================')
    parameters_path = f"checkpoints/{save_model_name}"
    print(f"Saving model parameters in {parameters_path}")
    evaluation_steps = num_steps_per_epoch // 1  # 5 evaluations per epoch

    if crossencoder:
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=test_evaluator,
            evaluation_steps=-1,
            warmup_steps=100,
            epochs=num_epochs,
            output_path=parameters_path,
            show_progress_bar=True,
            save_best_model=True,
            optimizer_params={"lr": learning_rate}
        )
    else:
        data_collator = DataCollatorWithTokenization(tok=tokenizer)
        training_args = TrainingArguments(output_dir=parameters_path, learning_rate=learning_rate, eval_steps=100,
                                          num_train_epochs=num_epochs, eval_strategy="steps", save_steps=200,
                                          remove_unused_columns=False, label_names=["labels"],
                                          per_device_train_batch_size=per_device_train_batch_size, logging_dir="./logs", logging_steps=100)
        import evaluate
        bin_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        score_metrics = evaluate.combine(["roc_auc"])
        def huggingface_eval(eval_pred):
            if isinstance(eval_pred.predictions, tuple):
                preds_scores = eval_pred.predictions[0]
            else:
                preds_scores = eval_pred.predictions
            preds_scores = softmax(preds_scores, axis=-1)
            metrics_dict = bin_metrics.compute(predictions=preds_scores[:,1]>0.5, references=eval_pred.label_ids)
            metrics_dict.update(score_metrics.compute(prediction_scores=preds_scores[:,1], references=eval_pred.label_ids))
            return metrics_dict

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=test_dataset[0],
            compute_metrics=huggingface_eval,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        #train_evaluator(model, output_path=parameters_path, epoch=ep)
        trainer.train()
        ## Call the legacy evaluator
        # test_evaluator(model, output_path=parameters_path, epoch=num_epochs)
        trainer.save_model(f"{parameters_path}/final/")
        parameters_path = f"{parameters_path}/final/"

    print(f'============================')
    print(f'==== Final evaluation ====')
    test_evaluator(model, output_path=".", epoch=num_epochs)
    print(f'============================')
    # model.save(parameters_path)
    upload_saved_model_to_s3(bucket_name=_BUCKET_NAME, local_path=parameters_path, s3_path=parameters_path)


if __name__ == "__main__":
    """
    TODO:
    - Save token limits for each model in 1k increments.
    """
    # model_name = 'cross-encoder/nli-deberta-v3-base' # base model, use 'vectara/hallucination_evaluation_model' if you want to further fine-tune ours"

    parser = argparse.ArgumentParser(
        prog="Harpo-hallucination-evaluation", description="", epilog=""
    )
    parser.add_argument("-d", "--dataset", choices=list(DATA_LOADER_FN.keys()))
    parser.add_argument("-t", "--test_dataset", choices=list(DATA_LOADER_FN.keys()), nargs="+")
    parser.add_argument("--save_model_name", default="")
    parser.add_argument("-m", "--base_model", choices=["cross-encoder/nli-deberta-v3-base", "vectara/hallucination_evaluation_model",
                                                       "allenai/longformer-base-4096", "tasksource/deberta-base-long-nli", "facebook/bart-large-mnli"], default="vectara/hallucination_evaluation_model")
    parser.add_argument("-c", "--label_col", type=str, default='label_binary')
    parser.add_argument("-e", "--num_epochs", type=int, default=3)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("--cuda_device_id", type=int, default=0)
    parser.add_argument("--train_layers", type=str, default=None)
    parser.add_argument("--save_params", action="store_true")

    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    print(args)

    dataset = DATA_LOADER_FN[args.dataset]()
    test_dataset = []
    for t in args.test_dataset:
        test_dataset.append(DATA_LOADER_FN[t]())

    #print(f"{COLOR.BOLD}Fine-tuning with {args.dataset} Dataset:{COLOR.ENDC}")
    print(f"Fine-tuning with Dataset: {args.dataset}")
    #print(f"{COLOR.BOLD}Parameters:{COLOR.ENDC}")
    print(f"\ttest dataset: {args.test_dataset}")
    print(f"\tsave_model_name: {args.save_model_name}")
    print(f"\tBase model: {args.base_model}")
    print(f"\tFine-tuning epochs: {args.num_epochs}")
    print(f"\tLearning rate: {args.learning_rate}")
    print(f"\tlabel_col: {args.label_col}")
    print(f"\tsave_params: {args.save_params}")

    fine_tune_train(
        dataset=dataset,
        test_dataset=test_dataset,
        base_model_name=args.base_model,
        save_model_name=args.save_model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        label_col=args.label_col,
        per_device_train_batch_size=args.batch_size,
        cuda_device_id=args.cuda_device_id,
        train_layers=args.train_layers.split(" ") if args.train_layers is not None else None,
        save_params=args.save_params,
        crossencoder=args.base_model in ["cross-encoder/nli-deberta-v3-base",
                                         "vectara/hallucination_evaluation_model",
                                         "allenai/longformer-base-4096"]
    )