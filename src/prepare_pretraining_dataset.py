from collections import defaultdict
from functools import partial
import logging
from multiprocessing import Process, Queue
import multiprocessing
import pickle
from sklearn.utils import gen_batches

from torch.multiprocessing import Pool
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
from tqdm import tqdm

from utils import set_random_seed


class MLMInstances(NamedTuple):
    tokens: torch.tensor
    masked_tokens: torch.tensor
    is_masked: torch.tensor
    segment_ids: torch.tensor
    is_random_next: torch.tensor

def combine_mlm_instances(batches:List[MLMInstances]):
    return MLMInstances(
        tokens = torch.cat([x.tokens for x in batches]),
        masked_tokens = torch.cat([x.masked_tokens for x in batches]),
        is_masked = torch.cat([x.is_masked for x in batches]),
        segment_ids = torch.cat([x.segment_ids for x in batches]),
        is_random_next = torch.cat([x.is_random_next for x in batches]),
    )


def apply_masking(
    tokens: torch.tensor,
    tokenizer,
    masked_ml_prob=0.15,
    replacement_probabilities=[0.8, 0.1, 0.1],
):
    is_masked = torch.bernoulli(torch.ones_like(tokens).float() * masked_ml_prob).bool()
    for special_token in [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    ]:
        is_masked &= tokens != special_token

    mask_replacement = torch.ones_like(tokens) * tokenizer.mask_token_id
    random_replacement = torch.randint(0, len(tokenizer), tokens.shape)

    replacement_choices = torch.multinomial(
        torch.tensor(replacement_probabilities), tokens.numel(), replacement=True
    ).long()
    replacement = torch.stack(
        [mask_replacement.flatten(), tokens.flatten(), random_replacement.flatten()]
    )[replacement_choices.long(), torch.arange(tokens.numel())].reshape(tokens.shape)

    masked_tokens = tokens.clone()
    masked_tokens[is_masked] = replacement[is_masked]

    return masked_tokens, is_masked

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Modified from Google's BERT repo."""
    tokens_a_interval = [0, len(tokens_a)]
    tokens_b_interval = [0, len(tokens_b)]
    while True:
        token_a_len = tokens_a_interval[1] - tokens_a_interval[0]
        token_b_len = tokens_b_interval[1] - tokens_b_interval[0]
        total_length = token_a_len + token_b_len
        if total_length <= max_num_tokens:
            break

        trunc_tokens = (
            tokens_a_interval if token_a_len > token_b_len else tokens_b_interval
        )
        assert trunc_tokens[0] < trunc_tokens[1]

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            trunc_tokens[0] += 1
        else:
            trunc_tokens[1] -= 1
    return (
        tokens_a[tokens_a_interval[0] : tokens_a_interval[1]],
        tokens_b[tokens_b_interval[0] : tokens_b_interval[1]],
    )

class PretrainingDatasetGenerator:
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer: PreTrainedTokenizer, documents: List[List[int]]):
        self.documents = documents
        self.tokenizer = tokenizer

    def get_instances(self, document, seq_len):
        instances_tokens = []
        instances_segment_ids = []
        instances_is_random_next = []

        document = document[::-1]
        max_num_tokens = seq_len - 3

        target_seq_len = max_num_tokens
        if random.random() < 0.1:
            target_seq_len = random.randint(2, max_num_tokens)

        while len(document):
            sampled_sentences = []
            while len(sum(sampled_sentences, [])) < target_seq_len and len(document):
                sampled_sentences.append(document.pop())

            num_a_sentences = random.randint(1, max(1,len(sampled_sentences)-1))
            part_a = sum(sampled_sentences[:num_a_sentences], [])
            sampled_sentences = sampled_sentences[num_a_sentences:]

            part_b = []
            if len(document) == 0 or random.random() < 0.5:
                is_random_next = True
                random_doc = self.documents[random.randint(0, len(self.documents) - 1)]
                for i in range(random.randint(0, len(random_doc) - 1), len(random_doc)):
                    part_b.extend(random_doc[i])
                    if len(part_a) + len(part_b) >= target_seq_len:
                        break
            else:
                is_random_next = False
                if len(sampled_sentences) == 0:
                    part_b = document.pop()
                else:
                    for i in range(len(sampled_sentences)):
                        part_b.extend(sampled_sentences[i])
                        if len(part_a) + len(part_b) >= target_seq_len:
                            sampled_sentences = sampled_sentences[i:]
                            break
            document += sampled_sentences[::-1]

            part_a, part_b = truncate_seq_pair(part_a, part_b, max_num_tokens)

            tokens = (
                [self.tokenizer.cls_token_id]
                + part_a
                + [self.tokenizer.sep_token_id]
                + part_b
                + [self.tokenizer.sep_token_id]
            )
            if len(tokens) < seq_len:
                tokens += [self.tokenizer.pad_token_id] * (seq_len - len(tokens))

            segment_ids = [0] * (len(part_a) + 2) + [1] * (seq_len - 2 - len(part_a))

            instances_tokens.append(tokens)
            instances_is_random_next.append(is_random_next)
            instances_segment_ids.append(segment_ids)

        tokens = torch.tensor(instances_tokens).long()
        segment_ids = torch.tensor(instances_segment_ids).long()
        is_random_next = torch.tensor(instances_is_random_next).bool()

        masked_tokens, is_masked = apply_masking(tokens, self.tokenizer)

        return MLMInstances(
            tokens=tokens.short(),
            masked_tokens=masked_tokens.short(),
            is_masked=is_masked.bool(),
            is_random_next=is_random_next.bool(),
            segment_ids=segment_ids.bool(),
        )

def generate_document_batch(args):
    logging.info("Start generate")
    documents, seq_len, seed = args
    set_random_seed(seed)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    dataset_generator = PretrainingDatasetGenerator(tokenizer, documents)
    logging.info("Generating...")
    gen_batches = [dataset_generator.get_instances(doc, seq_len) for doc in tqdm(documents)]
    gen_batch = combine_mlm_instances(gen_batches)
    logging.info(f"Generated {len(gen_batch.tokens)} instances")
    return pickle.dumps(gen_batch)

def prepare_pretraining_dataset(dataset_path,num_workers=8,seed=0):
    tokenized_documents = pickle.load(open(dataset_path, "rb"))
    logging.info("Loaded tokenized")
    batch_size = len(tokenized_documents) // num_workers + 1
    document_batches = [
        tokenized_documents[i : i + batch_size]
        for i in range(0, len(tokenized_documents), batch_size)
    ]
    seq_len = 128

    with Pool(num_workers) as p:
        res = list(p.imap_unordered(
            generate_document_batch,
            zip(
                document_batches,
                [seq_len] * num_workers,
                [seed]*num_workers
            ),
            chunksize=1
        ))
        logging.info("got all res")
        res = combine_mlm_instances([pickle.loads(x) for x in res])
    
    logging.info(f"Combined: {res.tokens.shape}")
    return res

def main():
    prepare_pretraining_dataset(Path("../wikipedia_tokenized/0"))

if __name__ == "__main__":
    main()
