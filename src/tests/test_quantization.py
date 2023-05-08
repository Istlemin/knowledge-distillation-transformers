import unittest
import copy
import torch
from transformers import BertConfig, BertForSequenceClassification
from utils import set_random_seed

from modeling.quantization import (
    MinMaxQuantizer,
    prepare_bert_for_quantization,
)
from utils import set_random_seed
from modeling.models import prepare_bert_for_kd

def tensor_equal_eps(a:torch.Tensor,b:torch.Tensor,eps:float = 1e-9):
    return ((a-b)<eps).all

class TestQuantisation(unittest.TestCase):
    def test_quantized_bert__32bit_same_as_original(self):
        """
        Assert that a 32-bit quantization gives same result as original model
        """

        config = BertConfig()

        model1 = BertForSequenceClassification(config)
        model2 = copy.deepcopy(model1)
        model2 = prepare_bert_for_quantization(
            model2,
            weight_quanter=MinMaxQuantizer(bits=32, clip_val=10000),
            act_quanter=MinMaxQuantizer(bits=32, clip_val=10000),
        )

        inp = torch.randint(200, 1000, (10, 30))
        set_random_seed(0)
        out1 = model1(
            inp, return_dict=True, output_hidden_states=True, output_attentions=True
        )
        set_random_seed(0)
        out2 = model2(
            inp, return_dict=True, output_hidden_states=True, output_attentions=True
        )

        print(torch.abs(out1.logits - out2.logits).max())
        self.assertLess(torch.abs(out1.logits - out2.logits).max(), 1e-5)

    def test_custom_attention(self):
        """
        Assert that model with custom attention layer replicate behaviour of transformers library:
        """

        config = BertConfig()

        model1 = BertForSequenceClassification(config)
        state_dict = model1.state_dict()
        model2 = prepare_bert_for_kd(BertForSequenceClassification(config))
        model2.load_state_dict(state_dict)

        inp = torch.randint(200, 1000, (10, 30))
        set_random_seed(0)
        out1 = model1(inp)
        set_random_seed(0)
        out2 = model2(inp)

        assert torch.all(out1.logits - out2.logits < 1e-6)

if __name__ == '__main__':
    unittest.main()
