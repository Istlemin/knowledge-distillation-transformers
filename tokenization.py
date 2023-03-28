import enum
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from datasets import Dataset

from datasets import load_dataset

from transformers import AutoTokenizer
import os
from tqdm import tqdm
import re
import pickle

def tokenize(dataset) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(sample):
        return tokenizer(
            sample["sentence"], max_length=128, padding="max_length", truncation=True
        )

    print("Tokenizing...")
    tokenized_dataset = dataset.filter(
        lambda sample: sample["sentence"] is not None
    ).map(tokenize_function)
    print("Num tokenized entries:", len(tokenized_dataset))
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
    tokenized_dataset["train"] = tokenized_dataset["train"].rename_column(
        "label", "labels"
    )
    tokenized_dataset["dev"] = tokenized_dataset["dev"].rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def split_document_into_sentences(document):
    return [
        sentence + "." for sentence in re.split(r"\.[\t\s\n]+", document) if sentence
    ]


def batched_prepare_datasets(document_dataset, outdir, batch_size=100000):
    shuffle_perm = list(range(len(document_dataset)))
    random.shuffle(shuffle_perm)
    batches = [
        document_dataset.select(shuffle_perm[i : i + batch_size])
        for i in tqdm(range(0, len(document_dataset), batch_size))
    ]

    for i,document_batch in enumerate(batches):
        documents = document_batch["text"]
        tokenized_documents = [
            [
                tokenizer.encode(sentence)[1:-1]
                for sentence in split_document_into_sentences(doc)
            ]
            for doc in tqdm(documents)
        ]
        pickle.dump(tokenized_documents, open(f"{i}","wb"))

    # processes = [
    #     Process(target=prepare_dataset, args=(batch, outdir / str(i)))
    #     for i, batch in enumerate(batches)
    # ]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()
    #     print("join!")


if __name__ == "__main__":
    dataset = load_dataset("wikipedia", "20220301.en")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    batch_size = len(dataset["train"]) // 64
    print("Tokenizing...")

    batched_prepare_datasets(dataset["train"], "../wikipedia_tokenized/", batch_size)
