import torch
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    AutoModelForMaskedLM,
    BertForMaskedLM,
)
from abc import abstractmethod


class ModelWithLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, input_ids, is_masked, output_ids):
        pass


def masked_lm_loss(model_outputs, input_ids, is_masked, output_ids):
    batch_size = len(input_ids)
    predictions = torch.argmax(
        model_outputs.logits * is_masked.reshape((batch_size, -1, 1)), dim=2
    )
    labels = output_ids * is_masked
    correct_predictions = torch.sum((predictions == labels) * is_masked).item()
    loss = torch.nn.functional.cross_entropy(
        model_outputs.logits.transpose(1, 2), output_ids
    )
    return loss, correct_predictions


class BertForMaskedLMWithLoss(ModelWithLoss):
    def __init__(self, model: BertForMaskedLM):
        super().__init__()

        self.model = model

    def forward(self, input_ids, is_masked, output_ids):
        outputs = self.model(input_ids, attention_mask=~is_masked)
        return masked_lm_loss(outputs, input_ids, is_masked, output_ids)


def get_bert_config(config_name):
    config: BertConfig = BertConfig.from_pretrained("bert-base-uncased", num_labels=5)

    if config_name == "base":
        return config
    if config_name == "tiny":
        config.num_attention_heads = 2
        config.intermediate_size = 512
        config.hidden_size = 128
        config.num_hidden_layers = 2
        return config


def load_pretrained_bert_base():
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5
    )


def load_untrained_bert_base():
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=5)
    return AutoModelForSequenceClassification.from_config(config)


def load_model_from_disk(path):
    return torch.load(path)
