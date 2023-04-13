from typing import Tuple
from datasets import Sequence
import torch
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    AutoModelForMaskedLM,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertPreTrainedModel,
    BertForPreTraining,
)
from abc import abstractmethod


class PretrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, input_ids: torch.tensor, is_masked: torch.tensor, output_ids: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        pass


class SequenceClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        attention_mask: torch.tensor,
        labels: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns
            loss: 0-dimensional float tensor
            predictions: 1-dimensional boolean tensor
        """
        pass


def pretraining_loss(model_outputs, output_ids, is_next_sentence, is_masked):
    word_correct_predictions = (
        model_outputs.prediction_logits.argmax(dim=-1) == output_ids
    )
    next_correct_predictions = (
        model_outputs.seq_relationship_logits.argmax(dim=-1) == is_next_sentence
    )
    loss_fct = torch.nn.CrossEntropyLoss()
    masked_lm_loss = loss_fct(
        model_outputs.prediction_logits.view(
            -1, model_outputs.prediction_logits.shape[-1]
        )[is_masked.flatten()],
        output_ids.view(-1)[is_masked.flatten()],
    )
    next_sentence_loss = loss_fct(
        model_outputs.seq_relationship_logits.view(-1, 2),
        is_next_sentence.view(-1).long(),
    )
    return (
        (masked_lm_loss + next_sentence_loss).reshape((1,)),
        word_correct_predictions,
        next_correct_predictions,
    )


class BertForPreTrainingWithLoss(PretrainingModel):
    def __init__(self, model: BertForPreTraining):
        super().__init__()

        self.model = model

    def forward(self, input_ids, is_masked, segment_ids, output_ids, is_next_sentence):
        outputs = self.model(
            input_ids,
            attention_mask=input_ids != 0,
            token_type_ids=segment_ids,
            return_dict=True,
            labels=output_ids,
            next_sentence_label=is_next_sentence.long(),
        )
        return pretraining_loss(outputs, output_ids, is_next_sentence, is_masked)

    def save(self, path):
        torch.save(self.model, path)


class BertForSequenceClassificationWithLoss(SequenceClassificationModel):
    def __init__(self, model: BertForMaskedLM):
        super().__init__()

        self.model = model

    def forward(self, **batch):
        outputs = self.model(**batch)
        predictions = torch.argmax(outputs.logits, dim=1)
        return outputs.loss, predictions

    def save(self, path):
        torch.save(self.model, path)


def get_bert_config(config_name):
    config: BertConfig = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)

    if config_name == "base":
        return config
    if config_name == "small12h":
        config.num_attention_heads = 12
        config.intermediate_size = 2048
        config.hidden_size = 504
        config.num_hidden_layers = 4
        return config
    if config_name == "small":
        config.num_attention_heads = 8
        config.intermediate_size = 2048
        config.hidden_size = 512
        config.num_hidden_layers = 4
        return config
    if config_name == "TinyBERT":
        config.num_attention_heads = 12
        config.intermediate_size = 1200
        config.hidden_size = 312
        config.num_hidden_layers = 4
        return config
    if config_name == "mini":
        config.num_attention_heads = 4
        config.intermediate_size = 1024
        config.hidden_size = 256
        config.num_hidden_layers = 4
        return config
    if config_name == "mini12h":
        config.num_attention_heads = 12
        config.intermediate_size = 1024
        config.hidden_size = 248
        config.num_hidden_layers = 4
        return config
    if config_name == "BERT_tiny":
        config.num_attention_heads = 2
        config.intermediate_size = 512
        config.hidden_size = 128
        config.num_hidden_layers = 2
        return config


def load_pretrained_bert_base():
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )


def load_untrained_bert_base():
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
    return AutoModelForSequenceClassification.from_config(config)


def load_model_from_disk(path):
    return torch.load(path)


def make_sequence_classifier(model: BertPreTrainedModel, num_labels):
    config = model.config
    config.num_labels = num_labels
    new_model = BertForSequenceClassification(config)
    state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
    new_model.load_state_dict(state_dict, strict=False)
    return new_model


if __name__ == "__main__":
    """
    Download models
    """

    pretrained_for_mlm = AutoModelForMaskedLM.from_config(get_bert_config("base"))
    torch.save(pretrained_for_mlm, "../models/pretrained_bert_mlm.pt")


def print_model(
    model: BertForSequenceClassification, input_ids, reps, attns, logits, grad=True
):
    print("Model weights:")
    print(model.bert.embeddings.word_embeddings.weight.view(torch.long).sum())
    print(
        model.bert.encoder.layer[0].attention.self.query.weight.view(torch.long).sum()
    )
    print(model.bert.encoder.layer[0].intermediate.dense.weight.view(torch.long).sum())
    print(
        model.bert.encoder.layer[-1].attention.self.query.weight.view(torch.long).sum()
    )
    print(model.bert.encoder.layer[-1].intermediate.dense.weight.view(torch.long).sum())
    print(model.bert.pooler.dense.weight.view(torch.long).sum())
    print(model.classifier.weight.view(torch.long).sum())

    if grad:
        print("Model grads:")
        print(model.bert.embeddings.word_embeddings.weight.grad.view(torch.long).sum())
        print(
            model.bert.encoder.layer[0]
            .attention.self.query.weight.grad.view(torch.long)
            .sum()
        )
        print(
            model.bert.encoder.layer[0]
            .intermediate.dense.weight.grad.view(torch.long)
            .sum()
        )
        print(
            model.bert.encoder.layer[-1]
            .attention.self.query.weight.grad.view(torch.long)
            .sum()
        )
        print(
            model.bert.encoder.layer[-1]
            .intermediate.dense.weight.grad.view(torch.long)
            .sum()
        )
        print(model.bert.pooler.dense.weight.grad.view(torch.long).sum())
        print(model.classifier.weight.grad.view(torch.long).sum())

    print("activations:")
    print(input_ids.shape)
    print(input_ids[0].view(torch.long).sum())
    print(reps[0].view(torch.long).sum())
    print(reps[1].view(torch.long).sum())
    print(reps[-1].view(torch.long).sum())
    print(attns[0].view(torch.long).sum())
    print(attns[1].view(torch.long).sum())
    print(attns[-1].view(torch.long).sum())
    print(logits.view(torch.long).sum())

