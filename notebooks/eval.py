import glob
import torch

from model import load_untrained_bert_base
from load_glue import load_tokenized_dataset, load_glue_sentence_classification
from finetune import run_epoch
from pathlib import Path


dataset = load_tokenized_dataset(
    Path("../../GLUE-baselines/glue_data/SST-2/"),
    load_glue_sentence_classification
)

def run_eval(model):
    loss, accuracy = run_epoch(model, torch.utils.data.DataLoader(
            dataset["dev"],
            batch_size=16,
        ), device="cpu")
    return loss, accuracy

from transformers import AutoModelForSequenceClassification
from model import get_bert_config

def print_checkpoints(path):
    checkpoints = sorted(glob.glob(str(path) + "checkpoint*"))
    for path in checkpoints:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        print(f"Epochs: {checkpoint['epochs']}:")
        print("Train loss:",checkpoint["train_loss"], "\t\tTrain accuracy:",checkpoint["train_accuracy"])
        print("Test loss:",checkpoint["test_loss"], "\t\tTest accuracy:",checkpoint["test_accuracy"])

        print(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"])

        #if checkpoint['epochs']==3:
        model = AutoModelForSequenceClassification.from_config(get_bert_config("base"))
        model = torch.nn.DataParallel(model)
        #print(checkpoint["model_state_dict"].keys())
        model.load_state_dict(checkpoint["model_state_dict"])
        print(run_eval(model))
        #torch.save(model,"../../models/finetuned_bert_base_ss2_92.4%.pt")


paths = glob.glob("../../checkpoints/*")
print(paths)

print_checkpoints(paths[0])