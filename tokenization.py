import enum
import multiprocessing
from pathlib import Path
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

def tokenize_document_batch(documents,outdir):
    print("Running process",outdir)
    tokenized_documents = [
        [
            tokenizer.encode(sentence)[1:-1]
            for sentence in split_document_into_sentences(doc)
        ]
        for doc in tqdm(documents)
    ]
    print("saving", outdir)
    pickle.dump(tokenized_documents, open(outdir,"wb"))

def batched_tokenize(document_dataset, outdir, batch_size=100000,num_workers=8):
    shuffle_perm = list(range(len(document_dataset)))
    random.shuffle(shuffle_perm)
    print("A")
    print("B")
    with multiprocessing.Pool(num_workers) as p:
        processes = []
        for ind,i in enumerate(tqdm(range(0, len(document_dataset), batch_size))):
            print(i)
            documents = document_dataset.select(shuffle_perm[i : i + batch_size])["text"]
            print("C")
            processes.append(p.apply_async(tokenize_document_batch,(documents,outdir/str(ind))))
        
        for proc in processes:
            proc.get()


if __name__ == "__main__":
    dataset = load_dataset("wikipedia", "20220301.en")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    batch_size = len(dataset["train"]) // 64
    print("Tokenizing...")

    batched_tokenize(dataset["train"], Path("../wikipedia_tokenized/"), batch_size)
