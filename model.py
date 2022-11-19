import torch
from transformers import AutoModelForSequenceClassification, BertConfig


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
    print("Loading model...")
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5
    )


def load_untrained_bert_base():
    print("Loading model...")
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=5)
    return AutoModelForSequenceClassification.from_config(config)


def load_model_from_disk(path):
    return torch.load(path)
