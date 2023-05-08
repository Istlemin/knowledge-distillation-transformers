import argparse
from pathlib import Path
from typing import List, Dict
from transformers import BertTokenizer, AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm
import random
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
    
from utils import set_random_seed
from load_glue import LOAD_FUNCTIONS


class GloveEmbeddings:
    embedding_vectors: np.array
    words: List[str]
    word_to_idx: Dict[str, int]

    def __init__(self, path: Path):
        self.embedding_vectors = []
        self.words = []
        print("Reading glove embeddings:")
        for line in tqdm(path.read_text().split("\n")[:-1]):
            line = line.split(" ")
            word = line[0]
            vector = np.array([float(x) for x in line[1:]])
            self.embedding_vectors.append(vector)
            self.words.append(word)
        self.embedding_vectors = np.array(self.embedding_vectors)
        self.embedding_vectors /= np.linalg.norm(self.embedding_vectors,axis=1).reshape((-1, 1))

        self.word_to_idx = {word: i for i, word in enumerate(self.words)}


def tokenized_to_words(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    words = []
    for token in tokens:
        if token[0] == "#":
            if len(words)==0:
                words.append([])
            words[-1].append(token)
        else:
            words.append([token])
    return words


def get_mlm_model_candidates(
    words,
    single_token_words,
    tokenizer: BertTokenizer,
    mlm_model,
    device,
    num_candidates,
    batch_size=64,
):
    if len(single_token_words) == 0:
        return []
    inputs = []
    masked_indices = []
    for i in single_token_words:
        words_with_masked = words.copy()
        words_with_masked[i] = ["[MASK]"]
        tokens_with_masked = sum(words, [])
        masked_index = len(sum(words[:i], [])) + 1

        input = tokenizer.convert_tokens_to_ids(tokens_with_masked)
        inputs.append([tokenizer.cls_token_id] + input)
        masked_indices.append(masked_index)

    inputs = torch.tensor(inputs)
    output_logits = torch.zeros(
        (inputs.shape[0], inputs.shape[1], tokenizer.vocab_size)
    )

    for i in range(0, len(inputs), batch_size):
        with torch.no_grad():
            output_logits[i : i + batch_size] = mlm_model(
                inputs[i : i + batch_size].to(device)
            ).logits.detach().cpu()  # shape is [sentences, tokens, vocab_size]

            torch.cuda.empty_cache()

    masked_predictions = output_logits[torch.arange(len(output_logits)), masked_indices]
    masked_topk = torch.topk(masked_predictions, num_candidates, dim=1).indices
    return [
        tokenizer.convert_ids_to_tokens(candidates)
        for candidates in masked_topk.tolist()
    ]


def get_glove_most_similar(word, glove_embeddings: GloveEmbeddings, num_candidates):
    if word not in glove_embeddings.word_to_idx:
        return [word]

    embedding = glove_embeddings.embedding_vectors[glove_embeddings.word_to_idx[word]]

    similarities = (
        glove_embeddings.embedding_vectors @ embedding.reshape((-1, 1))
    ).reshape((-1,))
    candidate_indices = torch.topk(torch.tensor(similarities), num_candidates).indices
    candidates = [glove_embeddings.words[i] for i in candidate_indices]
    return candidates


def wordpieces_to_word(wordpieces):
    return wordpieces[0] + "".join(suffix[2:] for suffix in wordpieces[1:])


def get_glove_candidates(
    words, words_for_glove, glove_embeddings: GloveEmbeddings, num_candidates
):
    return [
        get_glove_most_similar(
            wordpieces_to_word(words[i]), glove_embeddings, num_candidates
        )
        for i in words_for_glove
    ]


def augment_sentence(
    sentence,
    tokenizer: BertTokenizer,
    glove_embeddings: GloveEmbeddings,
    mlm_model,
    device,
    num_augmented=20,
    num_candidates=15,
    augment_prob=0.4,
    batch_size=8,
):
    words = tokenized_to_words(sentence, tokenizer)

    words = words[:64]

    words_for_glove = [i for i in range(len(words)) if len(words[i]) > 1]
    words_for_mlm_model = [i for i in range(len(words)) if len(words[i]) == 1]

    mlm_model_candidates = get_mlm_model_candidates(
        words, words_for_mlm_model, tokenizer, mlm_model, device, num_candidates
    )
    glove_candidates = get_glove_candidates(
        words, words_for_glove, glove_embeddings, num_candidates
    )

    candidates = [None for i in range(len(words))]
    for i, cand in zip(words_for_mlm_model, mlm_model_candidates):
        candidates[i] = cand
    for i, cand in zip(words_for_glove, glove_candidates):
        candidates[i] = cand

    augmented_sentences = []
    for i in range(num_augmented):
        augmented_words = words.copy()
        for j in range(len(words)):
            if random.uniform(0, 1) < augment_prob:
                augmented_words[j] = [random.choice(candidates[j])]
        augmented_sentences.append(
            " ".join(wordpieces_to_word(word) for word in augmented_words)
        )

    return augmented_sentences


def run_data_augmentation_batch(
    dataset,
    tokenizer: BertTokenizer,
    glove_embeddings: GloveEmbeddings,
    model_name,
    device,
):
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
    mlm_model.to(device)

    labels = []
    sentence = []
    sentence2 = []

    print("Augmenting dataset...")
    for i in tqdm(range(len(dataset.sentence))):
        augmented_sentences = augment_sentence(
            dataset.sentence[i], tokenizer, glove_embeddings, mlm_model, device
        )
        for augmented in augmented_sentences:
            sentence.append(augmented)
            labels.append(dataset.labels[i])

        if dataset.sentence2 is not None:
            augmented_sentences2 = augment_sentence(
                dataset.sentence2[i], tokenizer, glove_embeddings, mlm_model, device
            )
            for augmented in augmented_sentences2:
                sentence2.append(augmented)

    data_dict = {
        "labels":labels,
        "sentence": sentence,
    }
    if dataset.sentence2 is not None:
        data_dict["sentence2"] = sentence2
    
    return pd.DataFrame(data_dict)

def test(*args):
    print("getting model")
    mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    print("Running model")
    print("a",mlm_model(torch.tensor([[1,2,3]])))


def run_data_augmentation(
    dataset, glove_path, model_name="bert-base-uncased", num_processes=4,num_gpus=4
):  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    glove_embeddings = GloveEmbeddings(glove_path)   
    
    batch_size = len(dataset) // num_processes + 1

    batch_args = [
        (dataset[i*batch_size : (i+1)* batch_size], tokenizer, glove_embeddings, model_name, torch.device(f"cuda:{i%num_gpus}"))
        for i in range(0, num_processes)
    ]

    if num_processes == 1:
        augmented_dataset = run_data_augmentation_batch(*batch_args[0])
    else:
        with mp.Pool(num_processes) as p:
            augmented_dataset = p.starmap(run_data_augmentation_batch, batch_args)
        augmented_dataset = pd.concat(augmented_dataset)
    return augmented_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gluepath", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    set_random_seed(args.seed)
    
    dataset = LOAD_FUNCTIONS[args.dataset](args.gluepath / args.dataset).train[:1000]
    augmented_dataset = run_data_augmentation(dataset, args.glove, num_gpus=args.num_gpus, num_processes=args.num_processes)

    augmented_dataset.to_csv(args.gluepath / args.dataset / "train_aug.tsv", sep="\t")


if __name__ == "__main__":
    #mp.set_start_method('spawn',force=True)
    main()
