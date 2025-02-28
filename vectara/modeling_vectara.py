# Code from https://huggingface.co/vectara/hallucination_evaluation_model/blob/main/modeling_hhem_v2.py to allow finetuning the model.

import torch

# from peft import PeftModel
from transformers import PreTrainedModel, AutoConfig, T5ForTokenClassification, AutoModel, AutoTokenizer, \
    AutoModelForTokenClassification

from transformers import PretrainedConfig


class HHEMv2Config(PretrainedConfig):
    model_type = "HHEMv2"
    foundation = "google/flan-t5-base"
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"

    def __init___(self,
                  foundation="xyz",
                  prompt="abc",
                  **kwargs):
        super().__init__(**kwargs)
        self.foundation = foundation
        self.prompt = prompt


class HHEMv2Model(PreTrainedModel):
    config_class = HHEMv2Config

    def __init__(self, config):
        super().__init__(config)
    #     self.t5 = T5ForTokenClassification.from_config(
    #         AutoConfig.from_pretrained(config.foundation)
    #     )

    # def populate(self, model):
    #     self.t5 = model

    # def forward(self, **kwarg):
    #     return self.t5.transformer(**kwarg)


class HHEMv2ForSequenceClassification(PreTrainedModel):
    config_class = HHEMv2Config

    def __init__(self, config=HHEMv2Config()):
        super().__init__(config)
        self.t5 = T5ForTokenClassification(
            AutoConfig.from_pretrained(config.foundation)
        )
        self.prompt = config.prompt
        self.tokenzier = AutoTokenizer.from_pretrained(config.foundation)

    def populate(self, model: AutoModel):
        """Initiate the model with the pretrained model

        This method should only be called by Vectara employee who prepares the model for publishing. Users do not need to call this method.
        """
        self.t5 = model

    def forward(self, **kwargs):  # To cope with `text-classiication` pipeline
        self.t5.eval()
        with torch.no_grad():
            outputs = self.t5(**kwargs)
        logits = outputs.logits
        logits = logits[:, 0, :]
        outputs.logits = logits
        return outputs
        # return self.t5(**kwargs)

    def predict(self, text_pairs):
        tokenizer = self.tokenzier
        pair_dict = [{'text1': pair[0], 'text2': pair[1]} for pair in text_pairs]
        inputs = tokenizer(
            [self.prompt.format(**pair) for pair in pair_dict], return_tensors='pt', padding=True)
        self.t5.eval()
        with torch.no_grad():
            outputs = self.t5(**inputs)
        logits = outputs.logits
        logits = logits[:, 0, :]  # tok_cls
        transformed_probs = torch.softmax(logits, dim=-1)
        raw_scores = transformed_probs[:, 1]  # the probability of class 1
        return raw_scores
