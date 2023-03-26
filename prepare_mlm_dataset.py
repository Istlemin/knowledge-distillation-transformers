from collections import defaultdict
from functools import partial
from multiprocessing import Process, Queue
import multiprocessing
import pickle

# from pathos.multiprocessing import Pool
from datasets.arrow_dataset import Dataset
from matplotlib.lines import segment_hits
from transformers import PreTrainedTokenizer
from typing import List, Dict, NamedTuple
import torch
import random
import tqdm
import glob
from pathlib import Path
from datasets import concatenate_datasets
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from load_glue import load_batched_dataset
import argparse
import re


class PretrainItem(NamedTuple):
    tokens: torch.tensor
    masked_tokens: torch.tensor
    is_masked: torch.tensor
    segment_ids: torch.tensor
    is_random_next: torch.tensor


class PretrainingDatasetGenerator:
    mask_token: int
    tokenizer: PreTrainedTokenizer
    instances: List[Dict]
    is_wordpiece_suffix: torch.Tensor

    def __init__(self, tokenizer: PreTrainedTokenizer, documents: List[List[int]]):
        self.documents = documents  # [doc[::-1] for doc in documents]
        self.tokenizer = tokenizer
        assert self.tokenizer.mask_token_id is not None
        self.mask_token = self.tokenizer.mask_token_id
        self.is_wordpiece_suffix = torch.zeros(len(self.tokenizer), dtype=torch.bool)
        for token_string, idx in self.tokenizer.get_vocab().items():
            self.is_wordpiece_suffix[idx] = token_string[:2] == "##"
        self.vocab = list(self.tokenizer.get_vocab().values())
        self.special_tokens = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        ]

    def apply_masking(
        self,
        instance_tokens: List[int],
        seq_len: int,
        masked_ml_prob=0.15,
    ):
        masked_tokens = []
        is_masked = []

        for token in instance_tokens:
            if random.random() < masked_ml_prob and token not in self.special_tokens:
                r = random.random()
                if r < 0.8:
                    masked_tokens.append(self.tokenizer.mask_token_id)
                elif r < 0.9:
                    masked_tokens.append(token)
                else:
                    masked_tokens.append(random.choice(self.vocab))
                is_masked.append(True)
            else:
                masked_tokens.append(token)
                is_masked.append(False)
        return masked_tokens, is_masked

    def truncate_seq_pair(self,tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()
        return tokens_a, tokens_b

    def get_instances(self, document, seq_len):
        instances = []
        document = document[::-1]
        max_num_tokens = seq_len - 3

        target_seq_len = max_num_tokens
        if random.random() < 0.1:
            target_seq_len = random.randint(2, max_num_tokens)

        while len(document):
            sampled_sentences = []
            while len(sum(sampled_sentences, [])) < target_seq_len and len(document):
                sampled_sentences.append(document.pop())

            num_a_sentences = random.randint(1, len(sampled_sentences))
            part_a = sum(sampled_sentences[:num_a_sentences], [])
            sampled_sentences = sampled_sentences[num_a_sentences:]

            part_b = []
            if len(document) == 0 or random.random() < 0.5:
                is_random_next = True
                random_doc = self.documents[random.randint(0, len(self.documents) - 1)]
                for i in range(random.randint(0, len(random_doc) - 1), len(random_doc)):
                    part_b.extend(random_doc[i])
                    if len(part_a) + len(part_b) > target_seq_len:
                        break
            else:
                is_random_next = False
                if len(sampled_sentences)==0:
                    part_b = document.pop()
                else:
                    for i in range(len(sampled_sentences)):
                        part_b.extend(sampled_sentences[i])
                        if len(part_a) + len(part_b) > target_seq_len:
                            sampled_sentences = sampled_sentences[i:]
                            break
            document += sampled_sentences[::-1]
            
            part_a,part_b = self.truncate_seq_pair(part_a,part_b, max_num_tokens)

            tokens = (
                [self.tokenizer.cls_token_id]
                + part_a
                + [self.tokenizer.sep_token_id]
                + part_b
                + [self.tokenizer.sep_token_id]
            )
            if len(tokens)<seq_len:
                tokens += [self.tokenizer.pad_token_id] * (seq_len - len(tokens))
            segment_ids = [0]*(len(part_a)+2) + [1]*(seq_len-2-len(part_a))
            masked_tokens, is_masked = self.apply_masking(tokens, seq_len)

            instances.append(PretrainItem(
                tokens=tokens,
                masked_tokens=masked_tokens,
                is_masked=is_masked,
                is_random_next=is_random_next,
                segment_ids=segment_ids
            ))
        return instances

def transpose_dict(list_of_dicts: List[Dict]):
    res = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            res[k].append(v)
    return res


def prepare_dataset(document_dataset: Dataset, outdir: Path, seq_len=128):
    global dataset_generator

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset_generator = MLMDatasetGenerator(tokenizer)

    def document_to_instances(document):
        global dataset_generator
        return dataset_generator.doc_tokens_to_instances(
            torch.tensor(document["input_ids"]), seq_len=seq_len
        )

    # with Pool(16) as p:
    all_instances = map(
        document_to_instances, tqdm.tqdm(document_dataset)
    )  # , chunksize=100000
    dataset_dict = transpose_dict(sum(all_instances, []))
    for key in dataset_dict:
        dataset_dict[key] = torch.stack(dataset_dict[key])
    mlm_dataset = Dataset.from_dict(dataset_dict)
    mlm_dataset.save_to_disk(outdir)


def batched_prepare_datasets(document_dataset, outdir, batch_size=100000):
    shuffle_perm = list(range(len(document_dataset)))
    random.shuffle(shuffle_perm)
    batches = [
        document_dataset.select(shuffle_perm[i : i + batch_size])
        for i in tqdm.tqdm(range(0, len(document_dataset), batch_size))
    ]

    processes = [
        Process(target=prepare_dataset, args=(batch, outdir / str(i)))
        for i, batch in enumerate(batches)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        print("join!")


def split_document_into_sentences(document):
    return [
        sentence + "." for sentence in re.split(r"\.[\t\s\n]+", document) if sentence
    ]


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--in_dataset", dest="in_dataset_path", type=Path, required=True
    # )
    # parser.add_argument(
    #     "--out_dataset", dest="out_dataset_path", type=Path, required=True
    # )
    # args = parser.parse_args()

    # dataset = load_batched_dataset(args.in_dataset_path)

    # batched_prepare_datasets(dataset, Path(args.out_dataset_path))

    tokenizer = BertTokenizer.from_pretrained(
        "../models/bert_base_SST-2_93.58%/huggingface", do_lower_case=True
    )

    print("Reading file...")
    documents = Path("../data.txt").read_text().split("\n\n")

    print("Tokenizing...")

    # tokenized_documents = [[tokenizer.encode(sentence)[1:-1] for sentence in split_document_into_sentences(doc)] for doc in tqdm.tqdm(documents)]

    # pickle.dump(tokenized_documents, open("../data_tokenized","wb"))
    tokenized_documents = pickle.load(open("../data_tokenized", "rb"))

    dataset_generator = PretrainingDatasetGenerator(tokenizer, tokenized_documents)
    
    print("Go!")
    instances = []
    for doc in tokenized_documents:
        instances.extend(dataset_generator.get_instances(doc, 128))
    print(len(instances)/256)

    # for instance in instances:
    #     if instance.is_random_next == False:

if __name__ == "__main__":
    main()
