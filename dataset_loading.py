from pathlib import Path
import pandas

from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset

from tokenization import tokenize

def load_tsv(path):
    return Dataset.from_pandas(pandas.read_csv(path,sep="\t"))

def load_glue_sentence_classification(path):
    return DatasetDict(
        train=load_tsv(path / "train.tsv"),
        dev=load_tsv(path / "dev.tsv"),
        test=load_tsv(path / "test.tsv"),
    )

def load_tokenized_dataset(dataset_path:Path, dataset_load_function):
    tokenized_path = dataset_path / "tokenized"
    if tokenized_path.exists():
        return DatasetDict.load_from_disk(tokenized_path)
    else:
        datasets = dataset_load_function(dataset_path)
        datasets = tokenize(datasets)
        datasets.save_to_disk(tokenized_path)
        return datasets