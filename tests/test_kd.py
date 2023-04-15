import unittest
import torch
from transformers.modeling_outputs import MaskedLMOutput
from util import tensor_equal_eps

from kd import ConstantLayerMap, KDHiddenStates, LinearLayerMap
from model import get_bert_config

class TestPreparePretrainingDataset(unittest.TestCase):

    def test_apply_masking(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_linear_layer_map(self):
        student_layers = 3
        teacher_layers = 6

        layer_map = LinearLayerMap(3,6,initialisation=None)
        self.assertTrue(tensor_equal_eps(layer_map(torch.eye(6).float(), 0), torch.tensor([1/6,1/6,1/6,1/6,1/6,1/6]).float()))
        
        layer_map = LinearLayerMap(3,6,initialisation="uniform")
        self.assertTrue(tensor_equal_eps(layer_map(torch.eye(6).float(), 0), torch.tensor([0,1,0,0,0,0]).float()))
        self.assertTrue(tensor_equal_eps(layer_map(torch.eye(6).float(), 2), torch.tensor([0,0,0,0,0,1]).float()))
        
        layer_map = LinearLayerMap(3,6,initialisation="binned")
        self.assertTrue(tensor_equal_eps(layer_map(torch.eye(6).float(), 0), torch.tensor([1/2,1/2,0,0,0,0]).float()))
        self.assertTrue(tensor_equal_eps(layer_map(torch.eye(6).float(), 2), torch.tensor([0,0,0,0,1/2,1/2]).float()))

    def test_kd_hidden_state(self):
        teacher_cfg = get_bert_config("base")
        teacher_cfg.hidden_size = 12
        student_cfg = get_bert_config("TinyBERT")
        student_cfg.hidden_size = 12
        kd = KDHiddenStates(
            student_cfg,
            teacher_cfg,
            layer_map=ConstantLayerMap(
                student_cfg.num_hidden_layers,
                teacher_cfg.num_hidden_layers,
                map_type="uniform_start_0",
            ),
        )

        teacher_hidden = [torch.zeros((13,),dtype=torch.float) for _ in range(13)]
        for i in range(13):
            teacher_hidden[i][i] = i+1
        
        student_hidden = [torch.zeros((13,),dtype=torch.float) for _ in range(5)]
        student_hidden[0][0] = 1
        student_hidden[1][3] = 4
        student_hidden[2][6] = 7
        student_hidden[3][9] = 10
        student_hidden[4][12] = 13

        loss = kd(MaskedLMOutput(hidden_states=teacher_hidden),MaskedLMOutput(hidden_states=student_hidden))

        self.assertEqual(torch.tensor(0),loss)

if __name__ == '__main__':
    unittest.main()
