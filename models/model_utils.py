import json
import os
import random
from typing import Dict, Union
import numpy as np
import torch
from transformers import IntervalStrategy, TrainingArguments
import re
from dataset.dataset import LabellingMethods, TokenizerSpecialTokens
from enum import Enum

from models.behavior_attention import JoinOperation


class RatingPredictionModels(Enum):
    bert = "bert"  # also includes with behaviour features by passing number_turns_behaviour
    t5_enc = "t5_enc"  # also includes with behaviour features by passing number_turns_behaviour
    t5_enc_dec = "t5_enc_dec"  # only for text
    behaviour_only = "behaviour_only"  # includes models available in sklearn


class DefaultModelConfiguration:

    def __init__(self, configuration: Dict):
        # model and tokenizer
        self.suffix = configuration.get("suffix", "")  # way to distinguish between identical models
        self.model_name = configuration.get("model_name", "bert-base-uncased")
        self.tokenizer_name = configuration.get("tokenizer_name", "bert-base-uncased")
        self.truncation_side = configuration.get("truncation_side", "left")  # if left keeps the last part of a conversation
        self.model_type = configuration.get("model_type", RatingPredictionModels.bert.value)
        self.do_lower_case = configuration.get("do_lower_case", True)
        self.use_small_dataset = configuration.get("use_small_dataset", False)  # if true uses a small sample of the data for testing purposes

        # model specific (adjust these)
        self.pooling_layer = configuration.get("pooling_layer", True)
        # layer size applied to behaviour features (if None no layer is applied)
        self.linear_feature_layer_out = configuration.get("linear_feature_layer_out", None)  # type: int
        # layer size applied to the transformer output (if None no layer is applied)
        self.linear_transfomer_layer_out = configuration.get("linear_transfomer_layer_out", None)  # type: int
        self.activation = configuration.get("activation", None)  # str from Activation enum
        self.join_operation = configuration.get("join_operation", JoinOperation.concat.value)  # str from JoinOperation
        self.labelling_method = configuration.get("labelling_method", LabellingMethods.Binary_123_45.value)  # str from LabellingMethods
        self.scalling_method = configuration.get("scalling_method", None)  # ScalingMethod.StandardScalerMethod.value)
        self.number_turns_text = configuration.get("number_turns_text", 3)  # number of turns of text to consider
        self.number_turns_behaviour = configuration.get("number_turns_behaviour", 3)  # number of turns of behaviour to consider
        self.behaviour_features_size = configuration.get("behaviour_features_size", None)  # if none it will infer the size

        # special tokens to add (use None to not consider that token)
        self.system_separator_token = configuration.get("system_separator_token", TokenizerSpecialTokens.SYSTEM)
        self.user_separator_token = configuration.get("user_separator_token", TokenizerSpecialTokens.USER)
        self.replace_step_by_token = configuration.get("replace_step_by_token", TokenizerSpecialTokens.STEP)
        self.replace_curiosity_by_token = configuration.get("replace_curiosity_by_token", TokenizerSpecialTokens.CURIOSITY)
        # if true it adds a special token
        self.add_intent = configuration.get("add_intent", True)
        self.add_response_generator = configuration.get("add_response_generator", True)
        self.add_datasource = configuration.get("add_datasource", True)
        self.add_supports_screen = configuration.get("add_supports_screen", True)
        # if a number is provided it truncates the system responses to a max number of words
        self.max_system_words_per_turn = configuration.get("max_system_words_per_turn", 100)

        # training and eval params (check HuggingFace Trainer for more information)
        self.num_train_epochs = configuration.get("num_train_epochs", 20)  # total number of training epochs
        self.per_device_train_batch_size = configuration.get("per_device_train_batch_size", 16)
        self.per_device_eval_batch_size = configuration.get("per_device_eval_batch_size", 16)
        self.gradient_accumulation_steps = configuration.get("gradient_accumulation_steps", 1)
        self.warmup_steps = configuration.get("warmup_steps", 100 if self.use_small_dataset else 1000)
        self.learning_rate = configuration.get("learning_rate", 5e-5)
        self.weight_decay = configuration.get("weight_decay", 0.01)  # strength of weight decay
        self.logging_dir = configuration.get("logging_dir", './logs')  # directory for storing logs
        self.logging_steps = configuration.get("logging_steps", 50 if self.use_small_dataset else 100)
        self.evaluation_strategy = configuration.get("evaluation_strategy", IntervalStrategy.EPOCH.value)
        self.save_strategy = configuration.get("save_strategy", IntervalStrategy.EPOCH.value)
        # self.save_steps = configuration.get("save_steps",200 if self.use_small_dataset else 2000)
        self.save_total_limit = configuration.get("save_total_limit", 2)
        self.no_cuda = configuration.get("no_cuda", False)
        self.seed = configuration.get("seed", 42)
        self.metric_for_best_model = configuration.get("metric_for_best_model", None)
        self.greater_is_better = configuration.get("greater_is_better", None)

        # naming of the runs
        self.train_run_name = configuration.get("train_run_name", self.model_name.replace("/", "-")) + self.suffix
        self.test_run_name = configuration.get("test_run_name",
                                               self.model_name.replace("/", "-") + "_test") + self.suffix
        # number of samples to consider (if None it will use all the samples)
        self.number_samples = configuration.get("number_samples", 100 if self.use_small_dataset else None)

        # TODO set the data location
        self.data_location = configuration.get("data_location", ["./data/data.json"])
        self.start_date = configuration.get("start_date", None)  # in format e.g. "2022-06-01 00:00:00"
        self.end_date = configuration.get("end_date", None)  # in format e.g. "2022-06-01 00:00:00"
        self.min_num_turns = configuration.get("min_num_turns", 3)  # minimum number of turns to consider a conversation

        # output directory
        self.output_dir = configuration.get("output_dir", self.output_dir_from_params())

    def output_dir_from_params(self) -> str:
        out_dir_name = f'./rating_prediction_models/' \
                       f'{self.model_type}_{self.train_run_name}_{self.learning_rate}_{self.per_device_train_batch_size}'
        out_dir_name += f"_{self.labelling_method}_turnstext_{self.number_turns_text}_truncation_{self.truncation_side}"
        if self.number_turns_behaviour:
            out_dir_name += f"_behaviour_{self.number_turns_behaviour}_scalling_{self.scalling_method}"
            # extra layers
            extra_layer_str = ""
            extra_layer_str += f"_feature_{self.linear_feature_layer_out}" if self.linear_feature_layer_out else ""
            extra_layer_str += f"_transf_{self.linear_transfomer_layer_out}" if self.linear_transfomer_layer_out else ""
            extra_layer_str += f"_{self.activation}" if self.activation else ""
            extra_layer_str += f"_join_{self.join_operation}" if self.join_operation else ""
            out_dir_name += extra_layer_str

        if self.seed != 42:
            out_dir_name += f"_{self.seed}"
        return out_dir_name

    def get_trainer_training_arguments(self) -> TrainingArguments:
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            # save_steps=training_configuration.save_steps,
            save_total_limit=self.save_total_limit,
            no_cuda=self.no_cuda,
            seed=self.seed,
            run_name=self.train_run_name,
            load_best_model_at_end=True,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            report_to="none",
        )
        return training_args

    def get_trainer_testing_arguments(self) -> TrainingArguments:
        test_arguments = TrainingArguments(
            output_dir=self.output_dir,  # output directory
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            # batch size for evaluation
            logging_dir=self.logging_dir,  # directory for storing logs
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=IntervalStrategy.NO.value,
            no_cuda=False,
            seed=self.seed,
            run_name=self.test_run_name,
            report_to="none",
        )
        return test_arguments


class BehaviourOnlyModelConfiguration:

    def __init__(self, configuration: Dict):
        # model
        self.suffix = configuration.get("suffix", "")  # way to distinguish between identical models
        self.model_name = configuration.get("model_name", "behaviour_only")
        self.use_small_dataset = configuration.get("use_small_dataset", False)
        self.test_models = configuration.get("test_models", "all")  # or provide a list of str from BehaviourOnlyModels enum

        self.labelling_method = configuration.get("labelling_method", LabellingMethods.Binary_12_345.value)
        self.scalling_method = configuration.get("scalling_method", None)  # ScalingMethod.StandardScalerMethod.value)
        self.use_penultimate = configuration.get("use_penultimate", False)

        self.seed = configuration.get("seed", 42)
        self.train_run_name = configuration.get("train_run_name", self.model_name.replace("/", "-")) + self.suffix
        self.number_samples = configuration.get("number_samples", 100 if self.use_small_dataset else None)
        self.use_predefined_validation_split = configuration.get("use_predefined_validation_split", False)  # if true, use the predefined validation split else uses CV
        self.metric_for_best_model = configuration.get("metric_for_best_model", None)

        # TODO set the data location
        self.data_location = configuration.get("data_location", ["./data/data.json"])
        self.start_date = configuration.get("start_date", None)  # in format e.g. "2022-06-01 00:00:00"
        self.end_date = configuration.get("end_date", None)  # in format e.g. "2022-06-01 00:00:00"
        self.min_num_turns = configuration.get("min_num_turns", 3)  # minimum number of turns to consider a conversation

        # output directory
        self.output_dir = configuration.get("output_dir", "./rating_prediction_models/" + self.output_dir_from_params())
        self.visualization_dir = configuration.get("visualization_dir", "./visualizations/" + self.output_dir_from_params())

    def output_dir_from_params(self) -> str:
        penultimate_str = "penultimate" if self.use_penultimate else "last"
        valid_method = "" if self.use_predefined_validation_split else "cv"
        out_dir_name = f'{self.train_run_name}_{self.scalling_method}_{self.labelling_method}_' \
                       f'{penultimate_str}_{valid_method}'
        if self.seed != 42:
            out_dir_name += f"_{self.seed}"
        return out_dir_name


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)


def set_seeds(seed: int):
    # set all seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_checkpoint_name(text: str) -> str:
    # extract the name of the checkpoint given a str
    matches = re.findall(r"checkpoint-\d\d*", text)
    if matches:
        return matches[-1]  # if matches return the last match
    else:
        return ""  # if no matches return empty str


def class_attributes_to_dict(config_class: object) -> Dict:
    class_attributes = config_class.__dict__.items()
    json_attributes = {}
    for attr_name, attr_value in class_attributes:
        if attr_name not in ["__module__", "__dict__", "__weakref__", "__doc__"]:
            json_attributes[attr_name] = attr_value
    return json_attributes


def write_dict_to_json_file(data: Dict, output_path: str, indent: int = 2):
    with open(output_path, "w") as f_open:
        json.dump(data, f_open, indent=indent)


def write_configuration_to_file(config_class: object, output_path: str, indent: int = 2) -> Dict:
    json_attributes = class_attributes_to_dict(config_class)
    write_dict_to_json_file(json_attributes, f"{output_path}", indent)
    return json_attributes


def write_model_results_to_file(model_configuration: DefaultModelConfiguration,
                                checkpoint: Union[str, None] = None, metrics: Union[Dict, None] = None) -> Dict:
    if checkpoint:
        checkpoint = "_" + checkpoint
    else:
        checkpoint = ""

    # write the configuration used to a file for later use
    model_configuration_file_path = os.path.join(model_configuration.output_dir,
                                                 model_configuration.test_run_name + checkpoint + "_config.json")
    model_config_dict = write_configuration_to_file(config_class=model_configuration,
                                                    output_path=model_configuration_file_path)
    print(f"Model configuration written to file: {model_configuration_file_path}")
    print(model_config_dict)
    print()

    if metrics:
        model_metrics_file_path = os.path.join(model_configuration.output_dir,
                                               model_configuration.test_run_name + checkpoint + "_metrics.json")
        with open(model_metrics_file_path, "w") as f_open:
            print(f"Model metrics written to file: {model_metrics_file_path}")
            print(metrics)
            print()
            json.dump(metrics, f_open, indent=2)

        model_config_dict.update(metrics)

    return model_config_dict
