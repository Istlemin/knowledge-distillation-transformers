import pandas

from datasets import NamedSplit, load_dataset, Dataset, DatasetDict

def load_tsv(path):
    return Dataset.from_pandas(pandas.read_csv(path,sep="\t"))

def load_glue_sentence_classification(path):
    return DatasetDict(
        train=load_tsv(path / "train.tsv"),
        dev=load_tsv(path / "dev.tsv"),
        test=load_tsv(path / "test.tsv"),
    )