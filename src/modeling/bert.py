import copy
import math
import torch
from torch import nn
from typing import Optional, Tuple
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertSelfAttention

from modeling.quantization import (
    QuantizedEmbedding,
    QuantizedLinear,
    TwnQuantizer,
    MinMaxQuantizer,
)
from modeling.self_attention import CustomBertSelfAttention






def test_custom_attention():
    """
    Assert that kd model replicate behaviour of transformers library:
    """
    from transformers import BertForMaskedLM
    from utils import set_random_seed

    config = BertConfig()

    model1 = BertForMaskedLM(config)
    state_dict = model1.state_dict()
    model2 = prepare_bert_for_kd(BertForMaskedLM(config))
    model2.load_state_dict(state_dict)

    inp = torch.randint(200, 1000, (10, 30))
    set_random_seed(0)
    out1 = model1(inp)
    set_random_seed(0)
    out2 = model2(inp)

    assert torch.all(out1.logits - out2.logits < 1e-6)


if __name__ == "__main__":

    test_quantized_bert()
    test_custom_attention()

    print("Unit test passed")
