import torch
from transformers import AutoModelForSequenceClassification

def load_pretrained_bert_base():
    print("Loading model...")
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

def load_model_from_disk(path):
    return torch.load(path)