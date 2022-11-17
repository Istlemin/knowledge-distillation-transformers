from multiprocessing import Process, Queue
from pathos.multiprocessing import Pool
from datasets.arrow_dataset import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict
import torch
import random
import tqdm
import glob
from pathlib import Path
from datasets import concatenate_datasets
from transformers import AutoTokenizer


class MLMDatasetGenerator:
    mask_token: int
    tokenizer: PreTrainedTokenizer
    instances: List[Dict]
    is_wordpiece_suffix: torch.Tensor

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        assert self.tokenizer.mask_token_id is not None
        self.mask_token = self.tokenizer.mask_token_id
        self.is_wordpiece_suffix = torch.zeros(len(self.tokenizer), dtype=torch.bool)
        for token_string, idx in self.tokenizer.get_vocab().items():
            self.is_wordpiece_suffix[idx] = token_string[:2] == "##"

    def tokens_to_mlm_instance(
        self,
        instance_tokens: torch.tensor,
        seq_len: int,
        masked_ml_prob=0.15,
        replacement_probabilities=[0.8, 0.1, 0.1],
    ):
        in_len = len(instance_tokens)
        tokens = torch.ones(seq_len, dtype=torch.int32) * self.tokenizer.pad_token_id
        tokens[0] = self.tokenizer.cls_token_id
        tokens[1 : in_len + 1] = instance_tokens
        tokens[in_len + 1] = self.tokenizer.sep_token_id

        is_masked = torch.zeros(seq_len, dtype=torch.bool)
        is_masked[1 : in_len + 1] = torch.bernoulli(torch.ones(in_len) * 0.1)

        mask_replacement = torch.ones(seq_len, dtype=torch.int32) * self.mask_token
        self_replacement = (
            torch.ones(seq_len, dtype=torch.int32) * self.tokenizer.pad_token_id
        )
        self_replacement[1 : in_len + 1] = instance_tokens
        random_replacemet = torch.randint(0, len(self.tokenizer), (seq_len,))

        replacement_choices = torch.multinomial(
            torch.tensor(replacement_probabilities), seq_len, replacement=True
        )
        replacement = torch.stack(
            [mask_replacement, self_replacement, random_replacemet]
        )[replacement_choices, torch.arange(seq_len)].int()

        masked_tokens = tokens.clone()
        masked_tokens[is_masked] = replacement[is_masked]

        # words: List[List[int]] = []
        # for token in instance_tokens:
        #     if len(words) > 0 and self.is_wordpiece_suffix[token]:
        #         words[-1].append(token)
        #     else:
        #         words.append([token])

        # masked_tokens = []
        # is_masked = []
        # for word in words:
        #     if random.uniform(0, 1) < masked_ml_prob:  # Mask this word
        #         mask_replacement = [self.mask_token] * len(word)
        #         self_replacement = word
        #         random_replacement = [
        #             random.randint(0, len(self.tokenizer)) for _ in range(len(word))
        #         ]
        #         (replacement,) = random.choices(
        #             [mask_replacement, self_replacement, random_replacement],
        #             weights=replacement_probabilities,
        #         )
        #         masked_tokens += replacement
        #         is_masked += [True] * len(word)
        #     else:
        #         masked_tokens += word
        #         is_masked += [False] * len(word)

        # num_padding = in_len- seq_len
        # instance_tokens += [self.tokenizer.pad_token_id] * num_padding
        # masked_tokens += [self.tokenizer.pad_token_id] * num_padding
        # is_masked += [False] * num_padding

        return {
            "tokens": instance_tokens,
            "masked_tokens": masked_tokens,
            "is_masked": is_masked,
        }

    def doc_tokens_to_instances(self, doc_tokens: torch.tensor, seq_len: int):
        doc_tokens_per_seq = seq_len - 2
        return [
            self.tokens_to_mlm_instance(doc_tokens[i : i + doc_tokens_per_seq], seq_len)
            for i in range(0, len(doc_tokens), doc_tokens_per_seq)
        ]


def prepare_dataset(document_dataset: Dataset, q: Queue, seq_len=512):
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
    res = sum(list(all_instances), [])
    q.put(res)


def main():

    dataset_path = Path("../wikipedia_dataset/")
    dataset = concatenate_datasets(
        [
            Dataset.load_from_disk(path)
            for path in glob.glob(str(dataset_path / "tokenized*"))
        ]
    )

    batch_size = 100000
    batches = [
        dataset.select(range(i, i + batch_size))
        for i in tqdm.tqdm(range(0, len(dataset), batch_size))
    ]

    q = Queue()
    processes = [Process(target=prepare_dataset, args=(batch, q)) for batch in batches]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(len(q))


if __name__ == "__main__":
    main()
