#some helpers for the Long-Context Deberta model
from transformers import PreTrainedModel
from transformers import DebertaV2ForSequenceClassification, BartForSequenceClassification, T5ForSequenceClassification
import torch
from sentence_transformers import CrossEncoder
import numpy as np

class TwoWayDebertaV2(DebertaV2ForSequenceClassification):
    """ Transform three-way DeBERTaV2 into two-way model.
        This class is implemented so that it can be used as an Huggingface AutoModel,
        with support for checkpointing, loading etc.
    """

    def __init__(self, config, entail_label=None):
        """ Init the two way NLI Deberta. """
        #print(config)
        super().__init__(config)
        if entail_label is not None:
            self.entail_label = entail_label
        else:
            self.entail_label = config.label2id["entailment"]
        self.entail_mask = torch.zeros(3, dtype=torch.bool)
        self.entail_mask[self.entail_label] = True
        self.non_entail_mask = (self.entail_mask == False)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, **kvargs):
        in_dict = super().forward(**kvargs)
        ## Transform logits
        expsum = torch.log(torch.sum(torch.exp(in_dict.logits[:, self.non_entail_mask]), axis=1))
        in_dict.logits = torch.cat([expsum.reshape(-1, 1), in_dict.logits[:, self.entail_mask].reshape(-1, 1)], axis=1)
        ## Recompute loss!
        if "loss" in in_dict:
            # print("adapting loss")
            in_dict.loss = self.criterion(in_dict.logits, kvargs["labels"])
        return in_dict

class OneWayCrossEncoder(CrossEncoder):
    """ Transform three-way DeBERTaV2 into two-way model.
        This class is implemented so that it can be used as an Huggingface AutoModel,
        with support for checkpointing, loading etc.
    """

    def __init__(self, *vargs, **kvargs):
        """ Init the two way NLI Deberta. """
        #print(config)
        super().__init__(*vargs, **kvargs)
        self.entail_label = self.config.label2id["entailment"]
        self.entail_mask = torch.zeros(3, dtype=torch.bool)
        self.entail_mask[self.entail_label] = True
        self.non_entail_mask = (self.entail_mask == False)
        self.criterion = torch.nn.CrossEntropyLoss()

    def predict(self,  *vargs, **kvargs):
        in_dict = torch.tensor(super().predict(*vargs, **kvargs))
        ## Transform logits
        return torch.softmax(in_dict, dim=-1)[:, self.entail_mask].numpy().flatten()

class TwoWayBart(BartForSequenceClassification):
    """ Transform three-way DeBERTaV2 into two-way model.
        This class is implemented so that it can be used as an Huggingface AutoModel,
        with support for checkpointing, loading etc.
    """

    def __init__(self, config, entail_label=None):
        """ Init the two way NLI Deberta. """
        #print(config)
        super().__init__(config)
        if entail_label is not None:
            self.entail_label = entail_label
        else:
            self.entail_label = config.label2id["entailment"]
        self.entail_mask = torch.zeros(3, dtype=torch.bool)
        self.entail_mask[self.entail_label] = True
        self.non_entail_mask = (self.entail_mask == False)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, **kvargs):
        in_dict = super().forward(**kvargs)
        ## Transform logits
        expsum = torch.log(torch.sum(torch.exp(in_dict.logits[:, self.non_entail_mask]), axis=1))
        in_dict.logits = torch.cat([expsum.reshape(-1, 1), in_dict.logits[:, self.entail_mask].reshape(-1, 1)], axis=1)
        ## Recompute loss!
        if "loss" in in_dict:
            # print("adapting loss")
            in_dict.loss = self.criterion(in_dict.logits, kvargs["labels"])
        return in_dict

class TwoWayT5(T5ForSequenceClassification):
    """ Transform three-way DeBERTaV2 into two-way model.
        This class is implemented so that it can be used as an Huggingface AutoModel,
        with support for checkpointing, loading etc.
    """

    def __init__(self, config, entail_label=None):
        """ Init the two way NLI Deberta. """
        #print(config)
        super().__init__(config)
        if entail_label is not None:
            self.entail_label = entail_label
        else:
            self.entail_label = config.label2id["entailment"]
        self.entail_mask = torch.zeros(3, dtype=torch.bool)
        self.entail_mask[self.entail_label] = True
        self.non_entail_mask = (self.entail_mask == False)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, **kvargs):
        in_dict = super().forward(**kvargs)
        ## Transform logits
        expsum = torch.log(torch.sum(torch.exp(in_dict.logits[:, self.non_entail_mask]), axis=1))
        in_dict.logits = torch.cat([expsum.reshape(-1, 1), in_dict.logits[:, self.entail_mask].reshape(-1, 1)], axis=1)
        ## Recompute loss!
        if "loss" in in_dict:
            # print("adapting loss")
            in_dict.loss = self.criterion(in_dict.logits, kvargs["labels"])
        return in_dict

class DataCollatorWithTokenization:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        # print(batch)
        x = self.tok(list([[b["evidence"], b["claim"]] for b in batch]), return_tensors='pt', padding=True, truncation=False)
        x["labels"] = torch.tensor([bool(b["label_binary"]) for b in batch], dtype=torch.long)
        if len(batch) > 0 and "label_cert" in batch[0]:
            x["weight"] = torch.tensor([max(0, 1-2*np.abs(b["label_cert"]-b["label_binary"])) for b in batch])
        return x

class Vectara2DataCollatorWithTokenization:
    """ Data Collator for Vectara V2 """
    def __init__(self, tok, prompt):
        self.tok = tok
        self.prompt = prompt

    def __call__(self, batch):
        # print(batch)\
        pair_dict = [{'text1': b["evidence"], 'text2': b["claim"]} for b in batch]
        x = self.tok([self.prompt.format(**pair) for pair in pair_dict], return_tensors='pt', padding=True)
        x["labels"] = -100*torch.ones_like(x["input_ids"])
        x["labels"][:, 0] = torch.tensor([bool(b["label_binary"]) for b in batch], dtype=torch.long)
        if len(batch) > 0 and "label_cert" in batch[0]:
            x["weight"] = torch.tensor([max(0, 1-2*np.abs(b["label_cert"]-b["label_binary"])) for b in batch])
        return x