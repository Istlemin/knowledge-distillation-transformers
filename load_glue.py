from pathlib import Path
from typing import List, NamedTuple, Optional
import pandas as pd
import glob

from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from datasets import concatenate_datasets
from requests import head
from transformers import AutoTokenizer

import torch


class SCBatch(NamedTuple):
    input_ids: torch.tensor
    token_type_ids: torch.tensor
    attention_mask: torch.tensor
    labels: Optional[torch.tensor]


class SCDataset:
    def __init__(self, labels, sentence, sentence2=None):
        self.labels = labels
        self.sentence = sentence
        self.sentence2 = sentence2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return SCDataset(
            labels=None if self.labels is None else self.labels[item],
            sentence=None if self.sentence is None else self.sentence[item],
            sentence2=None if self.sentence2 is None else self.sentence2[item],
        )

    @staticmethod
    def from_df(df, label_col, sentence_col, sentence2_col=None):
        return SCDataset(
            None if label_col is None else df[label_col].tolist(),
            df[sentence_col].tolist(),
            None if sentence2_col is None else df[sentence2_col].tolist(),
        )


class SCDatasetTokenized(torch.utils.data.Dataset):
    def __init__(
        self, dataset: SCDataset, possible_labels: List[str], tokenizer_name="bert-base-uncased"
    ):

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if dataset.sentence2 is None:
            tokenizer_res = tokenizer(
                dataset.sentence,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokenizer_res = tokenizer(
                list(zip(dataset.sentence, dataset.sentence2)),
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        self.input_ids = tokenizer_res["input_ids"]
        self.token_type_ids = tokenizer_res["token_type_ids"]
        self.attention_mask = tokenizer_res["attention_mask"]
        self.labels = None
        if dataset.labels is not None:
            self.labels = torch.tensor(
                list(
                    possible_labels.index(label) if label in possible_labels else 0
                    for label in dataset.labels
                )
            )
        self.possible_labels = possible_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return SCBatch(
            labels=None if self.labels is None else self.labels[item],
            input_ids=self.input_ids[item],
            attention_mask=self.attention_mask[item],
            token_type_ids=self.token_type_ids[item],
        )


class SCDatasets:
    def __init__(self, train, dev, test, labels):
        self.train = train
        self.dev = dev
        self.test = test
        self.labels = labels


class SCDatasetsTokenized:
    def __init__(self, datasets: SCDatasets):
        self.train = SCDatasetTokenized(datasets.train, datasets.labels)
        self.dev = SCDatasetTokenized(datasets.dev, datasets.labels)
        self.test = SCDatasetTokenized(datasets.test, datasets.labels)

    def save_to_disk(self, path: Path):
        torch.save(self,path)

    @staticmethod
    def load_from_disk(path: Path):
        return torch.load(path)


def read_tsv(path, header="infer"):
    return pd.read_csv(
        path, sep="\t", header=header, error_bad_lines=False, keep_default_na=False, quotechar='', quoting=3
    )


def load_glue(
    path,
    augmented,
    labels,
    labels_col,
    sentence1_col,
    sentence2_col,
    train_file="train.tsv",
    dev_file="dev.tsv",
    test_file="test.tsv",
    header="infer",
    test_header=None,
    test_sentence1_col=None,
    test_sentence2_col=None,
):
    test_header = header if test_header is None else test_header
    test_sentence1_col = sentence1_col if test_sentence1_col is None else test_sentence1_col
    test_sentence2_col = sentence2_col if test_sentence2_col is None else test_sentence2_col

    if augmented:
        if sentence2_col is None:
            train = SCDataset.from_df(
                read_tsv(path / "train_aug.tsv"), "labels", "sentence"
            )
        else:
            train = SCDataset.from_df(
                read_tsv(path / "train_aug.tsv"), "labels", "sentence1", "sentence2"
            )
    else:
        train = SCDataset.from_df(
            read_tsv(path / train_file, header=header),
            labels_col,
            sentence1_col,
            sentence2_col,
        )

    dev = SCDataset.from_df(
        read_tsv(path / dev_file, header=header),
        labels_col,
        sentence1_col,
        sentence2_col,
    )
    test = SCDataset.from_df(
        read_tsv(path / test_file, header=test_header), None, test_sentence1_col, test_sentence2_col
    )
    return SCDatasets(train, dev, test, labels)

def load_wnli(path, augmented=False):
    return load_glue(
        path,
        augmented,
        labels_col="label",
        sentence1_col="sentence1",
        sentence2_col="sentence2",
        labels=[0,1],
    )

def load_qqp(path, augmented=False):
    return load_glue(
        path,
        augmented,
        labels_col="is_duplicate",
        sentence1_col="question1",
        sentence2_col="question2",
        labels=[0,1],
    )

def load_rte(path, augmented=False):
    return load_glue(
        path,
        augmented,
        labels_col="label",
        sentence1_col="sentence1",
        sentence2_col="sentence2",
        labels=["not_entailment","entailment"],
    )

def load_qnli(path, augmented=False):
    return load_glue(
        path,
        augmented,
        labels_col="label",
        sentence1_col="question",
        sentence2_col="sentence",
        labels=["not_entailment","entailment"],
    )

def load_mrpc(path, augmented=False):
    return load_glue(
        path,
        augmented,
        train_file="msr_paraphrase_train.txt",
        dev_file="msr_paraphrase_test.txt",
        test_file="test.tsv",
        labels_col="Quality",
        sentence1_col="#1 String",
        sentence2_col="#2 String",
        header="infer",
        labels=[0,1],
    )

def load_mnli(path, augmented=False):
    return load_glue(
        path,
        augmented,
        train_file="train.tsv",
        dev_file="dev_matched.tsv",
        test_file="test_matched.tsv",
        labels_col="gold_label",
        sentence1_col="sentence1",
        sentence2_col="sentence2",
        header="infer",
        labels=["contradiction", "entailment", "neutral"],
    )


def load_cola(path, augmented=False):
    return load_glue(
        path,
        augmented,
        labels_col=1,
        sentence1_col=3,
        sentence2_col=None,
        labels=[0,1],
        header=None,
        test_header="infer",
        test_sentence1_col="sentence",
        test_sentence2_col=None,
    )


def load_sst2(path, augmented=False):
    return load_glue(
        path,
        augmented,
        labels_col="label",
        sentence1_col="sentence",
        sentence2_col=None,
        labels=[0,1]
    )


LOAD_FUNCTIONS = {
    "CoLA": load_cola,
    "SST-2": load_sst2,
    "MNLI": load_mnli,
    "MRPC":load_mrpc,
    "QNLI":load_qnli,
    "QQP":load_qqp,
    "RTE":load_rte,
    "WNLI":load_wnli
}


def load_glue_dataset(glue_path, dataset_name, augmented=False):
    return LOAD_FUNCTIONS[dataset_name](glue_path / dataset_name, augmented)


def load_tokenized_glue_dataset(glue_path: Path, dataset_name, augmented=False):
    if augmented:
        tokenized_path = glue_path / dataset_name / "tokenized_aug"
    else:
        tokenized_path = glue_path / dataset_name / "tokenized"

    if tokenized_path.exists():
        return SCDatasetsTokenized.load_from_disk(tokenized_path)
    else:
        datasets = load_glue_dataset(glue_path, dataset_name, augmented=augmented)
        datasets = SCDatasetsTokenized(datasets)
        datasets.save_to_disk(tokenized_path)
        return datasets


def load_batched_dataset(dataset_path: Path):
    return concatenate_datasets(
        [Dataset.load_from_disk(path) for path in glob.glob(str(dataset_path / "*"))]
    )


if __name__ == "__main__":
    datasets = load_tokenized_glue_dataset(
        Path("../GLUE-baselines/glue_data/"), "MRPC", augmented=False
    )
    print(datasets)
    print(datasets.train[0])
