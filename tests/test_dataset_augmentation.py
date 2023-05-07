from pathlib import Path
from typing import Optional
import unittest
from transformers import AutoTokenizer, BertTokenizer, AutoModelForMaskedLM, BertForMaskedLM

from dataset_augmentation import GloveEmbeddings, augment_sentence, get_glove_most_similar, get_mlm_model_candidates, tokenized_to_words
from utils import set_random_seed

tokenizer : Optional[BertTokenizer] = None
glove_embeddings : Optional[GloveEmbeddings] = None
bert_mlm : Optional[BertForMaskedLM] = None

class TestDatasetAugmentation(unittest.TestCase):

    def test_tokenize_to_words__empty(self):
        sentence = ""
        tokenized_sentence = tokenized_to_words(sentence, tokenizer)
        
        self.assertEqual(tokenized_sentence, [])

    def test_tokenize_to_words__sentence(self):
        sentence = "I live in Stavanger."
        tokenized_sentence = tokenized_to_words(sentence, tokenizer)
        
        self.assertEqual(tokenized_sentence, [['i'], ['live'], ['in'], ['st', '##ava', '##nger'], ['.']])
    
    def test_glove_most_similar(self):
        res = get_glove_most_similar("small", glove_embeddings,5)
        
        self.assertEqual(res,['small', 'large', 'larger', 'smaller', 'tiny'])
    
    def test_glove_most_similar(self):
        res = get_glove_most_similar("small", glove_embeddings,5)
        
        self.assertEqual(res,['small', 'large', 'larger', 'smaller', 'tiny'])
    
    def test_get_mlm_model_candidates(self):
        words = [["this"],["is"],["cool"]]
        single_token_words = [0,1,2]

        candidates = get_mlm_model_candidates(
            words,
            single_token_words,
            tokenizer,
            mlm_model=bert_mlm,
            device="cpu",
            num_candidates=3,
        )

        self.assertEqual(candidates, [['all', 'this', 'it'], ['is', 'was', 'and'], ['cool', 'all', '"']])

    def test_augment_sentence__augment_prob_0(self):
        sentence = "I live in Stavanger"
        
        augmented = augment_sentence(
            sentence,
            tokenizer,
            glove_embeddings,
            bert_mlm,
            device="cpu",
            num_augmented=3,
            augment_prob=0,
        )
        
        self.assertEqual(augmented,['i live in stavanger', 'i live in stavanger', 'i live in stavanger'])

    def test_augment_sentence__augment_prob_stat(self):
        sentence = "I live in Stavanger"
        
        augmented = augment_sentence(
            sentence,
            tokenizer,
            glove_embeddings,
            bert_mlm,
            device="cpu",
            num_augmented=100,
            augment_prob=0.4,
        )

        total_augmented_words = 0

        for aug_sentence in augmented:
            total_augmented_words += sum(a!=b for a,b in zip(aug_sentence.split(" "), ['i','live','in','stavanger']))

        print(total_augmented_words)

        self.assertTrue(0.35<total_augmented_words/400<0.45)

    @classmethod
    def setUpClass(cls):
        global tokenizer, glove_embeddings, bert_mlm
        set_random_seed(0)
        glove_embeddings = GloveEmbeddings(Path("../glove.txt"))
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

if __name__ == '__main__':
    unittest.main()
