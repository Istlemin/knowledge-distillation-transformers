import logging
from typing import Dict, List, NamedTuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from model import ModelWithLoss, masked_lm_loss
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers import BertForMaskedLM, BertForSequenceClassification, BertConfig
from modeling.bert import prepare_bert_for_kd

from utils import set_random_seed


class KDLoss(nn.Module):
    pass


ModelOutput = Union[MaskedLMOutput, SequenceClassifierOutput]
Model = Union[BertForMaskedLM, BertForSequenceClassification]


def soft_cross_entropy(predicts, targets):
    # print("soft cross:")
    # print(predicts.flatten()[:5].tolist())
    # print(targets.flatten()[:5].tolist())
    predicts_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    loss = (-targets_prob * predicts_likelihood).mean()
    # print(loss.item())
    return loss

class KDPred(KDLoss):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        return soft_cross_entropy(student_output.logits, teacher_output.logits)


class KDAttention(KDLoss):
    def __init__(self, teacher_cfg: BertConfig, student_cfg: BertConfig, layer_map="uniform",):
        super().__init__()
        if layer_map=="uniform":
            # Evenly spread out layer map, 
            self.layer_map = [
                (i+1) * (teacher_cfg.num_hidden_layers) // (student_cfg.num_hidden_layers) - 1
                for i in range(student_cfg.num_hidden_layers)
            ]
        else:
            raise Exception("No such layer map implemented")

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        loss = 0 
        for student_att, teacher_att_layer in zip(student_output.attentions, self.layer_map):
            teacher_att = teacher_output.attentions[teacher_att_layer]
            student_att = student_att * (student_att > -1e2)
            teacher_att = teacher_att * (teacher_att > -1e2)
            loss += F.mse_loss(student_att, teacher_att)
        return loss

class KDHiddenStates(KDLoss):
    def __init__(self, teacher_cfg: BertConfig, student_cfg: BertConfig, layer_map="uniform"):
        super().__init__()

        if layer_map=="uniform":
            # Evenly spread out layer map
            self.layer_map = [
                i * (teacher_cfg.num_hidden_layers) // (student_cfg.num_hidden_layers)
                for i in range(student_cfg.num_hidden_layers + 1)
            ]
        else:
            raise Exception("No such layer map implemented")

        if student_cfg.hidden_size == teacher_cfg.hidden_size:
            self.student_to_teacher = None
        else:
            self.student_to_teacher = nn.Linear(student_cfg.hidden_size, teacher_cfg.hidden_size)
            # set_random_seed(0)
            # self.student_to_teacher.weight = torch.nn.Parameter((torch.rand((768, 312))*2-1)*0.05)
            # self.student_to_teacher.bias = torch.nn.Parameter(torch.rand((768,))*2-1)

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        loss = 0 
        for student_hidden, teacher_hidden_layer in zip(student_output.hidden_states, self.layer_map):
            teacher_hidden = teacher_output.hidden_states[teacher_hidden_layer]
            if self.student_to_teacher is None:
                loss += F.mse_loss(student_hidden, teacher_hidden)
            else:
                loss += F.mse_loss(self.student_to_teacher(student_hidden), teacher_hidden)
        return loss


class KDTransformerLayers(KDLoss):
    def __init__(self, teacher_cfg: BertConfig, student_cfg: BertConfig):
        super().__init__()

        self.kd_hidden_states = KDHiddenStates(teacher_cfg, student_cfg)
        self.kd_attention = KDAttention(teacher_cfg, student_cfg)

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        attention_loss = self.kd_attention(teacher_output, student_output)
        hidden_states_loss = self.kd_hidden_states(teacher_output, student_output)
        #print(f"att_loss: {attention_loss}, rep_loss: {hidden_states_loss}")
        return attention_loss + hidden_states_loss


class KD_MLM(ModelWithLoss):
    def __init__(self, teacher: Model, student: Model, kd_losses: List[KDLoss]):
        super().__init__()

        self.kd_losses = nn.ModuleList(kd_losses)
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.student = student

    def forward(self, input_ids, is_masked, output_ids):
        teacher_output = self.teacher.forward(
            input_ids,
            attention_mask=~is_masked,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        student_output = self.student.forward(
            input_ids,
            attention_mask=~is_masked,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )

        _, correct_predictions = masked_lm_loss(
            student_output, input_ids, is_masked, output_ids
        )

        loss = torch.zeros((1,), device=input_ids.device)
        for kd_loss in self.kd_losses:
            loss += kd_loss(teacher_output, student_output)
            # sprint(loss)
        return loss, correct_predictions

    def save(self,path):
        torch.save(self.student,path)

def print_model(model: BertForSequenceClassification, input_ids,reps,attns,logits, grad=True):
    print("Model weights:")
    print(model.bert.embeddings.word_embeddings.weight.flatten()[:5].tolist())
    print(model.bert.encoder.layer[0].attention.self.query.weight.flatten()[:5].tolist())
    print(model.bert.encoder.layer[0].intermediate.dense.weight.flatten()[:5].tolist())
    print(model.bert.encoder.layer[-1].attention.self.query.weight.flatten()[:5].tolist())
    print(model.bert.encoder.layer[-1].intermediate.dense.weight.flatten()[:5].tolist())
    print(model.bert.pooler.dense.weight.flatten()[:5].tolist())
    print(model.classifier.weight.flatten()[:5].tolist())

    if grad:
        print("Model grads:")
        print(model.bert.embeddings.word_embeddings.weight.grad.flatten()[:5].tolist())
        print(model.bert.encoder.layer[0].attention.self.query.weight.grad.flatten()[:5].tolist())
        print(model.bert.encoder.layer[0].intermediate.dense.weight.grad.flatten()[:5].tolist())
        print(model.bert.encoder.layer[-1].attention.self.query.weight.grad.flatten()[:5].tolist())
        print(model.bert.encoder.layer[-1].intermediate.dense.weight.grad.flatten()[:5].tolist())
        print(model.bert.pooler.dense.weight.grad.flatten()[:5].tolist())
        print(model.classifier.weight.grad.flatten()[:5].tolist())

    print("activations:")
    print(input_ids[0].flatten()[:5].tolist())
    print(reps[0].flatten()[:5].tolist())
    print(reps[-1].flatten()[:5].tolist())
    print(attns[0].flatten()[:5].tolist())
    print(attns[-1].flatten()[:5].tolist())
    print(logits.flatten()[:5].tolist())


class KD_SequenceClassification(ModelWithLoss):
    def __init__(
        self,
        teacher: Model,
        student: Model,
        kd_losses_dict: Dict[str, KDLoss],
        active_kd_losses: List[str],
    ):
        super().__init__()

        self.kd_losses = nn.ModuleDict(kd_losses_dict)
        self.active_kd_losses = active_kd_losses
        self.teacher = teacher 
        self.teacher.requires_grad_(False)
        self.student = student

    def forward(self, epoch=-1, **batch):
        #self.teacher.eval()
        #set_random_seed(0)
        teacher_output = self.teacher(
            return_dict=True, output_hidden_states=True, output_attentions=True, **batch
        )
        #set_random_seed(0)
        student_output = self.student(
            return_dict=True, output_hidden_states=True, output_attentions=True, **batch
        )
       
        predictions = torch.argmax(student_output.logits, dim=1)

        loss = torch.zeros((1,), device=batch["labels"].device)

        for kd_loss_name, kd_loss in self.kd_losses.items():
            curr_loss = kd_loss(teacher_output, student_output)

            if kd_loss_name in self.active_kd_losses:
                loss += curr_loss
            else:
                # hack to make sure all losses are part of loss computation,
                # needed for DistributedDataParallell
                loss += 0 * curr_loss
        
        return loss, predictions#, student_output, teacher_output
    
    def save(self,path):
        torch.save(self.student,path)
