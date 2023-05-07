import unittest
import torch
from transformers.modeling_outputs import MaskedLMOutput
from util import tensor_equal_eps

from kd import ConstantLayerMap, KDHiddenStates, LinearLayerMap
from model import get_bert_config

class TestQuantisation(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
