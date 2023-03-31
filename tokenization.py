from pathlib import Path
import random
from transformers import AutoTokenizer
from datasets import load_dataset

from transformers import AutoTokenizer
from tqdm import tqdm
import re
import pickle
import time

def split_document_into_sentences(document):
    return [
        sentence + "." for sentence in re.split(r"\.[\t\s\n]+", document) if sentence
    ]

def tokenize_document_batch(documents,outdir):
    print("Running process",outdir, time.time())
    tokenized_documents = [
        [
            sentence[1:-1]
            for sentence in tokenizer(split_document_into_sentences(doc))["input_ids"]
        ]
        for doc in tqdm(documents)
    ]
    print("saving", outdir, time.time())
    pickle.dump(tokenized_documents, open(outdir,"wb"))

def batched_tokenize(document_dataset, outdir, batch_size=100000,num_workers=8):
    shuffle_perm = list(range(len(document_dataset)))
    random.shuffle(shuffle_perm)

    for ind,i in enumerate(tqdm(range(0, len(document_dataset), batch_size))):
        print("Getting document batch")
        documents = document_dataset.select(shuffle_perm[i : i + batch_size])["text"]
        print("Tokenizing batch")
        tokenize_document_batch(documents,outdir/str(ind))


if __name__ == "__main__":
    dataset = load_dataset("wikipedia", "20220301.en")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    batch_size = len(dataset["train"]) // 64
    print("Tokenizing...")

    batched_tokenize(dataset["train"], Path("../wikipedia_tokenized/"), batch_size)
