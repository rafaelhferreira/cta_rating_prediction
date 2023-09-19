import torch
from transformers import BertModel, BertConfig, BertPreTrainedModel
from torch.nn import Linear
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from models.behavior_attention import BehaviouralAttention, model_calc_loss, JoinOperation, Activation


class BertWithBehaviourRatingPredictorConfig(BertConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert_pooling_layer = kwargs.get("bert_pooling_layer", True)
        self.linear_feature_layer_out = kwargs.get("linear_feature_layer_out", None)  # type: int
        self.linear_transfomer_layer_out = kwargs.get("linear_transfomer_layer_out", None)  # type: int
        self.activation = kwargs.get("activation", None)  # type: str
        self.join_operation = kwargs.get("join_operation", JoinOperation.concat.value)
        self.number_turns_behaviour = kwargs.get("number_turns_behaviour", None)
        self.behaviour_features_size = kwargs.get("behaviour_features_size", None)
        self.num_labels = kwargs.get("num_labels", 2)


class BertWithBehaviourRatingPredictor(BertPreTrainedModel):

    def __init__(self, config: BertWithBehaviourRatingPredictorConfig):

        super().__init__(config)
        self.number_turns = config.number_turns_behaviour
        self.behaviour_features_size = config.behaviour_features_size
        self.num_labels = config.num_labels

        self.bert_config = config
        self.rating_linear_in_features = self.bert_config.hidden_size

        self.bert_pooling_layer = config.bert_pooling_layer

        self.linear_feature_layer_out = config.linear_feature_layer_out
        self.linear_transfomer_layer_out = config.linear_transfomer_layer_out
        self.activation = config.activation
        if self.activation:
            if self.activation == Activation.tanh.value:
                self.activation = torch.tanh
            elif self.activation == Activation.relu.value:
                self.activation = torch.relu
            elif self.activation == Activation.sigmoid.value:
                self.activation = torch.sigmoid
            else:
                raise ValueError(f"Activation {self.activation} was not recognized.")

        self.join_operation = config.join_operation

        self.bert = BertModel(config, add_pooling_layer=self.bert_pooling_layer)

        self.behavioural_model = None
        self.number_turns_behaviour = config.number_turns_behaviour

        # if it has more than one behavior turn (will use attention) it should not have the other linear layer
        if self.number_turns_behaviour > 1:
            assert not self.linear_feature_layer_out, "If it has more than one behavior turn (will use attention) " \
                                                      "linear_feature_layer_out should be None"

        if self.number_turns_behaviour and self.behaviour_features_size:
            print("Loaded Attention Module")
            if self.number_turns_behaviour > 1:
                self.rating_linear_in_features += self.behaviour_features_size
                self.behavioural_model = BehaviouralAttention(turns=self.number_turns_behaviour,
                                                              features=self.behaviour_features_size)
            elif self.linear_feature_layer_out:  # if it is just one we use just a linear layer if active
                self.rating_linear_in_features += self.linear_feature_layer_out
                self.behavioural_model = Linear(in_features=self.behaviour_features_size,
                                                out_features=self.linear_feature_layer_out)
            else:
                self.rating_linear_in_features += self.behaviour_features_size
                print("Will use the raw feature vector")

        self.transformer_linear = None
        if self.linear_transfomer_layer_out:
            self.transformer_linear = Linear(in_features=self.bert_config.hidden_size,
                                             out_features=self.linear_transfomer_layer_out)
            # if linear over transformer and will concat we remove the hidden size and sum the out of the linear layer
            self.rating_linear_in_features = self.rating_linear_in_features - self.bert_config.hidden_size + self.linear_transfomer_layer_out

        # the previous rating_linear_in_features was done for concat so correct it here for sum and multiply
        if self.join_operation in [JoinOperation.sum.value, JoinOperation.multiply.value]:
            # if the operation is sum or multiply we use the self.linear_feature_layer_out as the input size
            if self.linear_feature_layer_out and not self.linear_transfomer_layer_out:
                assert self.linear_feature_layer_out == self.bert_config.hidden_size
                self.rating_linear_in_features = self.linear_feature_layer_out
            elif not self.linear_feature_layer_out and self.linear_transfomer_layer_out:
                assert self.linear_transfomer_layer_out == self.behaviour_features_size
                self.rating_linear_in_features = self.linear_transfomer_layer_out
            elif self.linear_feature_layer_out and self.linear_transfomer_layer_out:
                assert self.linear_feature_layer_out == self.linear_transfomer_layer_out
                self.rating_linear_in_features = self.linear_feature_layer_out
            else:
                raise ValueError("If join operation is sum or multiply it needs to match "
                                 "the features before the operation. Please check the model configuration.")

        self.dropout = nn.Dropout(p=0.3)
        self.rating_classifier = Linear(in_features=self.rating_linear_in_features, out_features=self.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                behaviour_features=None  # tensor (batch_size, number_turns, feature_size)
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]  # (batch_size, hidden_size)
        if self.transformer_linear:
            pooled_output = self.transformer_linear(pooled_output)
            if self.activation:
                pooled_output = self.activation(pooled_output)

        if self.behavioural_model is not None:
            behaviour_attended = self.behavioural_model(behaviour_features).squeeze(1)
            # only applies activation over linear because the other class already has a function
            if self.activation and isinstance(self.behavioural_model, Linear):
                behaviour_attended = self.activation(behaviour_attended)
        else:  # if there is no more than one turn there is no need to calculate attention
            behaviour_attended = behaviour_features.squeeze(1)

        if self.join_operation == JoinOperation.concat.value:
            joint_features = torch.cat((pooled_output, behaviour_attended), dim=1)
        elif self.join_operation == JoinOperation.sum.value:
            joint_features = torch.add(pooled_output, behaviour_attended)
        elif self.join_operation == JoinOperation.multiply.value:
            joint_features = torch.multiply(pooled_output, behaviour_attended)
        else:
            raise ValueError(f"JoinOperation {self.join_operation} not recognized.")

        features_dropout = self.dropout(joint_features)
        rating_logits = self.rating_classifier(features_dropout)
        loss = model_calc_loss(config=self.config, num_labels=self.num_labels,
                               labels=labels, rating_logits=rating_logits)
        if not return_dict:
            output = (rating_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=rating_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
