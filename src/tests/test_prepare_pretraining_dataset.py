from typing import Optional
import unittest
import torch
from transformers import AutoTokenizer, BertTokenizer

from prepare_pretraining_dataset import (
    PretrainingDatasetGenerator,
    apply_masking,
    truncate_seq_pair,
)
from utils import set_random_seed

tokenizer : Optional[BertTokenizer] = None

class TestPreparePretrainingDataset(unittest.TestCase):
    def test_apply_masking__masking_prob_0(self):
        tokens = torch.tensor([[1, 2], [3, 4]])

        masked_tokens, is_masked = apply_masking(tokens, tokenizer, masked_ml_prob=0)

        self.assertTrue(torch.equal(tokens, masked_tokens))
        self.assertTrue(torch.all(is_masked == False))

    def test_apply_masking__masking_prob_1(self):
        tokens = torch.tensor([[1, 2], [3, 4]])

        masked_tokens, is_masked = apply_masking(
            tokens, tokenizer, masked_ml_prob=1, replacement_probabilities=[1.0, 0, 0]
        )

        self.assertTrue(torch.all(masked_tokens == tokenizer.mask_token_id))
        self.assertTrue(torch.all(is_masked == True))

    def test_apply_masking__dont_mask_special(self):
        tokens = torch.tensor(
            [
                [
                    tokenizer.cls_token_id,
                    tokenizer.sep_token_id,
                    tokenizer.pad_token_id,
                ]
            ]
        )

        masked_tokens, is_masked = apply_masking(tokens, tokenizer, masked_ml_prob=1)

        self.assertTrue(torch.equal(masked_tokens, tokens))
        self.assertTrue(torch.all(is_masked == False))

    def test_apply_masking__original_if_not_masked(self):
        tokens = torch.ones((100, 1000)) * 2000

        masked_tokens, is_masked = apply_masking(tokens, tokenizer)

        self.assertTrue(torch.all((masked_tokens == tokens) | is_masked))

    def test_apply_masking__statistic(self):
        mask_prob = 0.15
        replace_mask_prob = 0.8
        replace_original_prob = 0.1

        tokens = torch.ones((100, 1000)) * 2000
        size = tokens.numel()

        masked_tokens, is_masked = apply_masking(tokens, tokenizer)
        num_replaced_with_mask = (
            (masked_tokens == tokenizer.mask_token_id) & is_masked
        ).sum()
        num_replaced_with_original = ((masked_tokens == tokens) & is_masked).sum()

        def assert_close(val, target):
            print(val, target)
            self.assertTrue(target * 0.9 < val < target * 1.1)

        assert_close(is_masked.sum(), size * mask_prob)
        assert_close(num_replaced_with_mask, size * mask_prob * replace_mask_prob)
        assert_close(
            num_replaced_with_original, size * mask_prob * replace_original_prob
        )

    def test_truncate_seq_pair__b_longer(self):
        tokens_a = [0] * 10
        tokens_b = [0] * 15
        tokens_a, tokens_b = truncate_seq_pair(tokens_a, tokens_b, 20)

        self.assertEqual(len(tokens_a), 10)
        self.assertEqual(len(tokens_b), 10)

    def test_truncate_seq_pair__a_longer(self):
        tokens_a = [0] * 15
        tokens_b = [0] * 10
        tokens_a, tokens_b = truncate_seq_pair(tokens_a, tokens_b, 20)

        self.assertEqual(len(tokens_a), 10)
        self.assertEqual(len(tokens_b), 10)

    def test_truncate_seq_pair__takes_middle(self):
        tokens_a = list(range(100))
        tokens_b = list(range(100))
        tokens_a, tokens_b = truncate_seq_pair(tokens_a, tokens_b, 100)

        self.assertTrue(all(a + 1 == b for a, b in zip(tokens_a, tokens_a[1:])))
        self.assertTrue(all(a + 1 == b for a, b in zip(tokens_b, tokens_b[1:])))
        self.assertTrue(10 < tokens_a[0])
        self.assertTrue(tokens_a[-1] < 90)
        self.assertTrue(10 < tokens_b[0])
        self.assertTrue(tokens_b[-1] < 90)

    def test_get_instances__right_format(self):
        gen = PretrainingDatasetGenerator(tokenizer, [[[2, 2, 2, 2]]])
        tokens = gen.get_instances([[1, 1, 1, 1]], 11).tokens.tolist()

        self.assertEqual(tokens, [[101, 1, 1, 1, 1, 102, 2, 2, 2, 2, 102]])

    def test_get_instances__use_all(self):
        gen = PretrainingDatasetGenerator(tokenizer, [[[2, 2, 2, 2], [4, 4, 4, 4]]])

        tokens = gen.get_instances([[1, 1, 1, 1], [3, 3, 3, 3]], 11).tokens.tolist()

        self.assertEqual(len(tokens), 2)
        self.assertEqual(len(tokens[0]), 11)
        self.assertEqual(len(tokens[1]), 11)
        self.assertEqual(sum(x==1 for x in sum(tokens,[])),4)
        self.assertEqual(sum(x==3 for x in sum(tokens,[])),4)

    @classmethod
    def setUpClass(cls):
        global tokenizer
        set_random_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


if __name__ == "__main__":
    unittest.main()
