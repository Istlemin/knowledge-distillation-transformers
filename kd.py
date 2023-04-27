from typing import Dict, List, Union

import torch
from torch import nn
import torch.nn.functional as F

from model import PretrainingModel, SequenceClassificationModel, pretraining_loss
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers import BertForMaskedLM, BertForSequenceClassification, BertConfig



ModelOutput = Union[MaskedLMOutput, SequenceClassifierOutput]
Model = Union[BertForMaskedLM, BertForSequenceClassification]

class KDLoss(nn.Module):
    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput) -> torch.tensor:
        pass

class LayerMap(nn.Module):
    def forward(self, teacher_data : torch.tensor, layer_index:int) -> torch.tensor:
        pass

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



class ConstantLayerMap(LayerMap):
    def __init__(self, student_layers, teacher_layers, map_type="uniform"):
        super().__init__()
        if map_type == "uniform_start_0":
            # Evenly spread out layer map,
            # with student layer 0 mapped to teacher layer 0
            self.layer_map = [
                i * (teacher_layers) // (student_layers)
                for i in range(student_layers + 1)
            ]
        elif map_type == "uniform":
            # Evenly spread out layer map
            self.layer_map = [
                (i + 1) * (teacher_layers) // (student_layers) - 1
                for i in range(student_layers)
            ]
        elif map_type == "beginning":
            # Map student the first layers of teacher
            self.layer_map = list(range(student_layers + 1))
        elif map_type == "end":
            # Map student the first layers of teacher
            self.layer_map = list(range(student_layers + 1))
        else:
            raise Exception("No such layer map implemented")

    def forward(self, teacher_data, layer_index):
        return teacher_data[self.layer_map[layer_index]]


class LinearLayerMap(LayerMap):
    def __init__(self, student_layers, teacher_layers, initialisation=None):
        super().__init__()
        self.layer_map = torch.ones((student_layers, teacher_layers))
        if initialisation == "uniform":
            for i in range(student_layers):
                self.layer_map[
                    i, (i + 1) * (teacher_layers) // (student_layers) - 1
                ] = 10000000
        elif initialisation == "binned":
            for i in range(teacher_layers):
                self.layer_map[i * (student_layers) // (teacher_layers), i] = 10000000
        elif initialisation is not None:
            raise Exception("No such layer map implemented")
        self.layer_map = nn.Parameter(self.layer_map)

    def forward(self, teacher_data, layer_index):
        lmap = F.softmax(self.layer_map[layer_index], dim=0).reshape(
            (-1,) + tuple(1 for _ in range(len(teacher_data[0].shape)))
        )
        return torch.sum(teacher_data * lmap.reshape((-1, 1)), dim=0)


class KDAttention(KDLoss):
    def __init__(self, teacher_cfg: BertConfig,student_cfg: BertConfig, layer_map:LayerMap):
        super().__init__()
        self.layer_map = layer_map

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        teacher_atts = torch.stack(teacher_output.attentions)
        teacher_atts = (teacher_atts > -1e2) * teacher_atts

        loss = 0
        for i, student_att in enumerate(student_output.attentions):
            student_att = student_att * (student_att > -1e2)
            teacher_att = self.layer_map(teacher_atts, i)
            loss += F.mse_loss(student_att, teacher_att)
        return loss


class KDHiddenStates(KDLoss):
    def __init__(
        self, teacher_cfg: BertConfig, student_cfg:BertConfig, layer_map, transform_per_layer=False
    ):
        super().__init__()

        self.layer_map = layer_map

        student_layers = student_cfg.num_hidden_layers + 1
        if student_cfg.hidden_size == teacher_cfg.hidden_size:
            self.student_to_teacher = [None] * student_layers
        else:
            if transform_per_layer:
                self.student_to_teacher = nn.ModuleList(
                    [
                        nn.Linear(student_cfg.hidden_size, teacher_cfg.hidden_size)
                        for i in range(student_layers)
                    ]
                )
            else:
                self.student_to_teacher = nn.ModuleList(
                    [nn.Linear(student_cfg.hidden_size, teacher_cfg.hidden_size)]
                    * student_layers
                )

            # set_random_seed(0)
            # self.student_to_teacher.weight = torch.nn.Parameter((torch.rand((768, 312))*2-1)*0.05)
            # self.student_to_teacher.bias = torch.nn.Parameter(torch.rand((768,))*2-1)

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        loss = 0
        for i, (student_to_teacher, student_hidden) in enumerate(
            zip(self.student_to_teacher, student_output.hidden_states)
        ):
            teacher_hidden = self.layer_map(
                torch.stack(teacher_output.hidden_states), i
            )
            if student_to_teacher is None:
                loss += F.mse_loss(student_hidden, teacher_hidden)
            else:
                loss += F.mse_loss(student_to_teacher(student_hidden), teacher_hidden)
        return loss


class KDTransformerLayers(KDLoss):
    def __init__(self, teacher_cfg: BertConfig, student_cfg: BertConfig):
        super().__init__()

        self.kd_hidden_states = KDHiddenStates(
            teacher_cfg,
            student_cfg,
            layer_map=ConstantLayerMap(
                student_cfg.num_hidden_layers,
                teacher_cfg.num_hidden_layers,
                map_type="uniform_start_0",
            ),
        )
        self.kd_attention = KDAttention(
            teacher_cfg,
            student_cfg,
            layer_map=ConstantLayerMap(
                student_cfg.num_hidden_layers,
                teacher_cfg.num_hidden_layers,
                map_type="uniform",
            ),
        )

    def forward(self, teacher_output: ModelOutput, student_output: ModelOutput):
        attention_loss = self.kd_attention(teacher_output, student_output)
        hidden_states_loss = self.kd_hidden_states(teacher_output, student_output)
        return attention_loss + hidden_states_loss


class KDPreTraining(PretrainingModel):
    def __init__(self, teacher: Model, student: Model, kd_losses: List[KDLoss]):
        super().__init__()

        self.kd_losses = nn.ModuleList(kd_losses)
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.student = student

    def forward(self, input_ids, is_masked, segment_ids, output_ids, is_next_sentence):
        teacher_output = self.teacher.forward(
            input_ids,
            attention_mask=input_ids != 0,
            token_type_ids=segment_ids,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        student_output = self.student.forward(
            input_ids,
            attention_mask=input_ids != 0,
            token_type_ids=segment_ids,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
            labels=output_ids,
            next_sentence_label=is_next_sentence.long(),
        )

        (
            pretrain_loss,
            word_correct_predictions,
            next_correct_prediction,
        ) = pretraining_loss(student_output, output_ids, is_next_sentence, is_masked)

        loss = 0
        for kd_loss in self.kd_losses:
            loss += kd_loss(teacher_output, student_output)

        return (
            torch.stack([loss[0], pretrain_loss]),
            word_correct_predictions,
            next_correct_prediction,
        )

    def save(self, path):
        torch.save(self.student, path)


class KDSequenceClassification(SequenceClassificationModel):
    def __init__(self, teacher: Model, student: Model, kd_losses: List[KDLoss]):
        super().__init__()

        self.kd_losses = nn.ModuleList(kd_losses)
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.student = student

    def forward(self, **batch):
        self.teacher.eval()

        student_output = self.student(
            return_dict=True, output_hidden_states=True, output_attentions=True, **batch
        )
        teacher_output = self.teacher(
            return_dict=True, output_hidden_states=True, output_attentions=True, **batch
        )

        predictions = torch.argmax(student_output.logits, dim=1)

        loss = 0

        for kd_loss in self.kd_losses:
            loss += kd_loss(teacher_output, student_output)

        return loss, predictions

    def save(self, path):
        torch.save(self.student, path)
