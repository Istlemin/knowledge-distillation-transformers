import torch
from transformers import AutoModelForSequenceClassification, BertConfig

def load_pretrained_bert_base():
    print("Loading model...")
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

def load_untrained_bert_base():
    print("Loading model...")
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=5)
    return AutoModelForSequenceClassification.from_config(config)


def load_model_from_disk(path):
    return torch.load(path)