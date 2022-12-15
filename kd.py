from typing import List, NamedTuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from model import ModelWithLoss, masked_lm_loss
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers import BertForMaskedLM, BertForSequenceClassification, BertConfig


class KDLoss(nn.Module):
    pass


ModelOutput = Union[MaskedLMOutput, SequenceClassifierOutput]
Model = Union[BertForMaskedLM, BertForSequenceClassification]


class KDPred(KDLoss):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        return F.mse_loss(teacher_output.logits, student_output.logits) / 10


class KDAttention(KDLoss):
    def __init__(self, layer_map):
        super().__init__()
        self.layer_map = layer_map

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        teacher_attentions = torch.stack(teacher_output.attentions)
        student_attentions = torch.stack(student_output.attentions)
        return F.mse_loss(teacher_attentions[self.layer_map], student_attentions)


class KDHiddenStates(KDLoss):
    def __init__(self, layer_map, teacher_cfg: BertConfig, student_cfg: BertConfig):
        super().__init__()
        self.layer_map = layer_map
        self.student_to_teacher = nn.ModuleList(
            [
                nn.Linear(student_cfg.hidden_size, teacher_cfg.hidden_size)
                for i in range(len(layer_map))
            ]
        )

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        teacher_hidden_states = torch.stack(teacher_output.hidden_states)
        student_hidden_states = torch.stack(student_output.hidden_states)
        return torch.stack(
            [
                F.mse_loss(
                    self.student_to_teacher[i](student_hidden_states[i]),
                    teacher_hidden_states[self.layer_map[i]],
                )
                for i in range(len(self.layer_map))
            ]
        ).mean()


class KDTransformerLayers(KDLoss):
    def __init__(self, teacher_cfg: BertConfig, student_cfg: BertConfig):
        super().__init__()

        # Evenly spread out layer map
        layer_map = [
            i * (teacher_cfg.num_hidden_layers) // (student_cfg.num_hidden_layers)
            for i in range(student_cfg.num_hidden_layers)
        ]
        self.kd_hidden_states = KDHiddenStates(layer_map, teacher_cfg, student_cfg)
        self.kd_attention = KDAttention(layer_map)

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        attention_loss = self.kd_attention(teacher_output, student_output)
        hidden_states_loss = self.kd_hidden_states(teacher_output, student_output)
        # print(attention_loss * 1000, hidden_states_loss)
        return attention_loss * 100 + hidden_states_loss * 0.00001


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


class KD_SequenceClassification(ModelWithLoss):
    def __init__(self, teacher: Model, student: Model, kd_losses: List[KDLoss]):
        super().__init__()

        self.kd_losses = nn.ModuleList(kd_losses)
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.student = student

    def forward(self, **batch):
        teacher_output = self.teacher(**batch)
        student_output = self.student(**batch)

        predictions = torch.argmax(student_output.logits, dim=1)
        correct_predictions += torch.sum(predictions == batch["labels"])

        loss = torch.zeros((1,), device=batch["labels"].device)
        for kd_loss in self.kd_losses:
            loss += kd_loss(teacher_output, student_output)
            # sprint(loss)
        return loss, correct_predictions
