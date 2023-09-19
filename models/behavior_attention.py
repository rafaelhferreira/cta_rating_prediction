import torch
from torch import nn
from torch.nn import Linear, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import PretrainedConfig
from enum import Enum


class Activation(Enum):
    tanh = "tanh"
    relu = "relu"
    sigmoid = "sigmoid"


# operation that is applied to combine behaviour and conversational-flow features
class JoinOperation(Enum):
    concat = "concat"
    sum = "sum"
    multiply = "multiply"


class TrainableEltwiseLayer(nn.Module):

    def __init__(self, in_features):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, in_features))  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        return x * self.weights  # element-wise multiplication


class BehaviouralAttention(nn.Module):

    def __init__(self, turns, features):
        super(BehaviouralAttention, self).__init__()

        self.behaviour_linear = Linear(in_features=features, out_features=1)
        self.context_vector = TrainableEltwiseLayer(in_features=turns)
        nn.init.uniform_(self.context_vector.weights)

    def forward(self, b_1):
        scores = self.behaviour_linear(b_1)
        scores_tanh = torch.tanh(scores)
        score_vs_context = self.context_vector(scores_tanh.squeeze())
        attention_weights = torch.softmax(score_vs_context, dim=1)
        features_multiplied = torch.bmm(attention_weights.unsqueeze(dim=1), b_1)
        return features_multiplied


def model_calc_loss(config: PretrainedConfig, num_labels: int,
                    labels: torch.Tensor, rating_logits: torch.Tensor):
    # calculates the loss
    loss = None
    # from transformers
    if labels is not None:
        if config.problem_type is None:
            if num_labels == 1:
                config.problem_type = "regression"
            elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                config.problem_type = "single_label_classification"
            else:
                config.problem_type = "multi_label_classification"

        if config.problem_type == "regression":
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(rating_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(rating_logits, labels)
        elif config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(rating_logits.view(-1, num_labels), labels.view(-1))
        elif config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(rating_logits, labels)

    return loss
