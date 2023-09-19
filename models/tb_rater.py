import json
import os
from collections import Counter
from typing import Tuple, Set, Union, List, Dict
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, max_error, mean_squared_error, \
    mean_absolute_error, r2_score
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoConfig, \
    AutoModelForSequenceClassification, Trainer, T5ForConditionalGeneration
from data_binding.interactions import AllInteractions
from dataset.dataset import labeling_method_to_number_of_labels, TokenizerSpecialTokens, RatingPredictionDataset, \
    LabellingMethods, convert_conversation_label, count_occurrences, split_dataset, convert_str_to_number_repr, \
    DATASET_SEED
from models.bert_based import BertWithBehaviourRatingPredictorConfig, BertWithBehaviourRatingPredictor
from models.model_utils import DefaultModelConfiguration, class_attributes_to_dict, set_seeds, extract_checkpoint_name, \
    write_model_results_to_file, RatingPredictionModels
from models.session_features import ScalingMethod, SessionFeaturesVectors
from models.t5_based import T5EncoderWithBehaviourRatingPredictor, T5WithBehaviourRatingPredictorConfig, \
    T5ConfigForSequenceClassification, T5EncoderForSequenceClassification


def load_model_tokenizer(model_configuration: DefaultModelConfiguration) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_configuration.tokenizer_name,
                                              do_lower_case=model_configuration.do_lower_case,
                                              truncation_side=model_configuration.truncation_side)
    number_labels = labeling_method_to_number_of_labels(model_configuration.labelling_method)
    if model_configuration.number_turns_behaviour:  # use the custom model
        model_config = AutoConfig.from_pretrained(model_configuration.model_name).to_dict()

        if model_configuration.model_type == RatingPredictionModels.bert.value:
            config = BertWithBehaviourRatingPredictorConfig(
                bert_pooling_layer=model_configuration.pooling_layer,
                linear_feature_layer_out=model_configuration.linear_feature_layer_out,
                linear_transfomer_layer_out=model_configuration.linear_transfomer_layer_out,
                activation=model_configuration.activation,
                join_operation=model_configuration.join_operation,
                number_turns_behaviour=model_configuration.number_turns_behaviour,
                behaviour_features_size=model_configuration.behaviour_features_size,
                num_labels=number_labels,
                **model_config)
            model = BertWithBehaviourRatingPredictor.from_pretrained(model_configuration.model_name, config=config)

        elif model_configuration.model_type == RatingPredictionModels.t5_enc.value:
            config = T5WithBehaviourRatingPredictorConfig(
                linear_feature_layer_out=model_configuration.linear_feature_layer_out,
                linear_transfomer_layer_out=model_configuration.linear_transfomer_layer_out,
                activation=model_configuration.activation,
                join_operation=model_configuration.join_operation,
                number_turns_behaviour=model_configuration.number_turns_behaviour,
                behaviour_features_size=model_configuration.behaviour_features_size,
                num_labels=number_labels,
                **model_config)
            model = T5EncoderWithBehaviourRatingPredictor.from_pretrained(model_configuration.model_name, config=config)
        else:
            raise NotImplementedError(f"Model type: {model_configuration.model_type} is not implemented.")
    else:  # load the other models
        if model_configuration.model_type == RatingPredictionModels.t5_enc.value:
            # since t5 does not have a sequence classification version load a custom one
            model_config = AutoConfig.from_pretrained(model_configuration.model_name).to_dict()
            config = T5ConfigForSequenceClassification(num_labels=number_labels, **model_config)
            model = T5EncoderForSequenceClassification.from_pretrained(model_configuration.model_name, config=config)
        elif model_configuration.model_type == RatingPredictionModels.t5_enc_dec.value:
            tokenizer = AutoTokenizer.from_pretrained(model_configuration.tokenizer_name)
            model = T5ForConditionalGeneration.from_pretrained(model_configuration.model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_configuration.model_name,
                                                                       num_labels=number_labels)
    return model, tokenizer


def add_tokens_to_tokenizer(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, tokens_to_add: Set[str]):
    # add the tokens
    tokens_list = list(tokens_to_add)
    tokens_list.sort()  # sort the tokens for repr
    number_of_added_tokens = tokenizer.add_tokens(tokens_list)
    print("Number tokens: ", len(tokens_to_add))
    print("Number of tokens added: ", number_of_added_tokens)
    print()

    # resize the model due to the added tokens
    model.resize_token_embeddings(len(tokenizer))


def handle_special_token_addition(tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                                  model_configuration: DefaultModelConfiguration,
                                  intents: Union[List[str], None] = None,
                                  response_gens: Union[List[str], None] = None):
    # add special tokens to tokenizer if needed before creating the dataset
    tokens_to_add = set()
    if model_configuration.system_separator_token:
        tokens_to_add.add(model_configuration.system_separator_token)
    if model_configuration.user_separator_token:
        tokens_to_add.add(model_configuration.user_separator_token)
    if model_configuration.replace_step_by_token:
        tokens_to_add.add(model_configuration.replace_step_by_token)
    if model_configuration.replace_curiosity_by_token:
        tokens_to_add.add(model_configuration.replace_curiosity_by_token)
    if model_configuration.add_datasource:
        tokens_to_add.update([TokenizerSpecialTokens.RECIPE, TokenizerSpecialTokens.WIKIHOW,
                              TokenizerSpecialTokens.MULTISOURCE])
    if model_configuration.add_supports_screen:
        tokens_to_add.update([TokenizerSpecialTokens.SCREEN, TokenizerSpecialTokens.NO_SCREEN])
    if model_configuration.add_intent:
        if not intents:
            raise ValueError("model_configuration.add_intent passed as True, but no intents were passed to function.")
        else:
            tokens_to_add.update([TokenizerSpecialTokens.str_to_special_token_converter(intent) for intent in intents])
    if model_configuration.add_response_generator:
        if not response_gens:
            raise ValueError("model_configuration.add_response_generator passed as True, but no RG "
                             "were passed to function.")
        else:
            tokens_to_add.update([TokenizerSpecialTokens.str_to_special_token_converter(rg) for rg in response_gens])

    if tokens_to_add:
        add_tokens_to_tokenizer(tokenizer=tokenizer, model=model, tokens_to_add=tokens_to_add)


def compute_metrics_rating_prediction_classification(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        "precision": precision_score(y_pred=predictions, y_true=labels, average="macro"),
        "recall": recall_score(y_pred=predictions, y_true=labels, average="macro"),
        "f1": f1_score(y_pred=predictions, y_true=labels, average="macro"),
        "accuracy": accuracy_score(y_pred=predictions, y_true=labels)
    }
    return metrics


def compute_metrics_rating_prediction_regression(eval_pred):
    logits, labels = eval_pred

    metrics = {
        "max_error": max_error(y_pred=logits, y_true=labels),
        "mean_squared_error": mean_squared_error(y_pred=logits, y_true=labels),
        "mean_absolute_error": mean_absolute_error(y_pred=logits, y_true=labels),
        "r2_score": r2_score(y_pred=logits, y_true=labels)
    }
    return metrics


def compute_metrics_rating_prediction_seq2seq_classification(eval_pred):
    (logits, _), labels = eval_pred

    # get the index of the first token predicted
    predictions = np.argmax(logits[:, 0, :], axis=-1)
    class_label = labels[:, 0]

    metrics = {
        "precision": precision_score(y_pred=predictions, y_true=class_label, average="macro"),
        "recall": recall_score(y_pred=predictions, y_true=class_label, average="macro"),
        "f1": f1_score(y_pred=predictions, y_true=class_label, average="macro"),
        "accuracy": accuracy_score(y_pred=predictions, y_true=class_label)
    }
    return metrics


def get_predictions(evaluator: Trainer, tokenizer: PreTrainedTokenizer,
                    number_labels: int, test_dataset: RatingPredictionDataset,
                    labelling_method: LabellingMethods,
                    write_predictions_to_file: str = None, print_to_console: bool = False,
                    is_text_2_text: bool = False) -> Tuple[Dict, List]:
    evaluator.model.eval()
    with torch.no_grad():
        prediction_output = evaluator.predict(test_dataset=test_dataset)
    predictions_dict = {}
    all_predictions, all_ratings = [], []

    if is_text_2_text:
        # [0] because when it is text_2_text it also returns the encoder_last_hidden_state
        iterable_predictions = prediction_output.predictions[0]
    else:
        iterable_predictions = prediction_output.predictions

    for i in tqdm(range(len(iterable_predictions)), desc="Predictions"):
        prediction = torch.tensor(iterable_predictions[i])
        if number_labels != 1:
            if not is_text_2_text:
                prediction = torch.argmax(torch.softmax(prediction, dim=-1), dim=-1)
                prediction = prediction.item()
                prediction_repr = prediction
            else:
                prediction = torch.argmax(prediction[0], dim=-1).item()  # first token is the label
                prediction = tokenizer.convert_ids_to_tokens(prediction)
                prediction_repr = f"{convert_str_to_number_repr(prediction)} / {prediction}"  # visualization purposes
        else:  # is regression
            prediction = prediction.item()
            prediction_repr = prediction

        current_session = test_dataset.sessions[i]
        if print_to_console:
            print()
            print("#################################")
            print(current_session.session_id)
            print(current_session)
            print("Rating:", current_session.rating)
            print("Rating Prediction:", prediction_repr)
            print("#################################")
            print()

        predictions_dict[current_session.session_id] = {
            "original_rating": current_session.rating,
            "label_rating": convert_conversation_label(method=labelling_method, label_value=current_session.rating),
            "predicted_rating_repr": prediction_repr,
            "predicted_rating": prediction,
            "session": current_session.session_to_simple_dict(),
        }

        all_predictions.append(prediction)
        all_ratings.append(convert_conversation_label(method=labelling_method, label_value=current_session.rating))

    print("Prediction General Stats:")
    print("Ratings Distribution:", count_occurrences(all_ratings))
    print("Predictions Distribution:", count_occurrences(all_predictions))
    print()

    if write_predictions_to_file:
        with open(write_predictions_to_file, "w") as f_open:
            json.dump(predictions_dict, f_open, indent=2)

    return predictions_dict, all_predictions


def train_eval_model(configuration: Dict):
    # train model and evaluate on validation and test sets

    # get the model configuration
    model_configuration = DefaultModelConfiguration(configuration)
    print("Model Configuration:")
    print(class_attributes_to_dict(model_configuration))
    print()

    # load the data
    set_seeds(DATASET_SEED)  # the seed for the dataset is always the same
    all_sessions = AllInteractions(file_paths=model_configuration.data_location,
                                   min_num_turns=model_configuration.min_num_turns,
                                   only_rated=True, number_samples=model_configuration.number_samples,
                                   start_date=model_configuration.start_date, end_date=model_configuration.end_date)

    labelling_method = LabellingMethods(model_configuration.labelling_method)
    scalling_method = model_configuration.scalling_method
    if model_configuration.scalling_method:
        scalling_method = ScalingMethod(model_configuration.scalling_method)

    s_train, s_val, s_test, l_train, l_val, l_test = split_dataset(
        sessions=list(all_sessions.sessions_dict_obj.values()),
        labelling_method=labelling_method,
    )

    # we want to use user behaviour turns but the feature size is not specified
    if model_configuration.number_turns_behaviour and model_configuration.behaviour_features_size is None:
        # infer the size of the features by using a single example from the training set
        feature_names, _ = SessionFeaturesVectors.get_features_from_turn(session=s_train[0],
                                                                         turn_index=s_train[0].get_number_dialogue_turns()-1)
        # update the model configuration with that size
        model_configuration.behaviour_features_size = len(feature_names)

    # set seeds for reproducibility
    set_seeds(model_configuration.seed)
    # get model and tokenizer
    model, tokenizer = load_model_tokenizer(model_configuration=model_configuration)

    # add special tokens
    handle_special_token_addition(tokenizer=tokenizer, model=model, model_configuration=model_configuration,
                                  intents=all_sessions.get_all_intents(),
                                  response_gens=all_sessions.get_all_response_generators())

    # update configuration to keep it consistent
    custom_config_dict = class_attributes_to_dict(model_configuration)
    final_dict = {}
    for k, v in custom_config_dict.items():
        if model.config and k not in model.config.to_dict():  # avoid putting on top of an existing value
            final_dict[k] = v
    model.config.update(final_dict)

    is_text_2_text = model_configuration.model_type == RatingPredictionModels.t5_enc_dec.value
    number_labels = labeling_method_to_number_of_labels(model_configuration.labelling_method)

    # create the datasets using the tokenized data and labels
    print("Creating Train Dataset")
    train_dataset = RatingPredictionDataset(tokenizer=tokenizer, sessions=s_train, labels_list=l_train,
                                            num_labels=number_labels,
                                            number_turns_behaviour=model_configuration.number_turns_behaviour,
                                            scaling=scalling_method,
                                            number_turns_text=model_configuration.number_turns_text,
                                            system_separator_token=model_configuration.system_separator_token,
                                            user_separator_token=model_configuration.user_separator_token,
                                            max_system_words_per_turn=model_configuration.max_system_words_per_turn,
                                            replace_step_by_token=model_configuration.replace_step_by_token,
                                            replace_curiosity_by_token=model_configuration.replace_curiosity_by_token,
                                            add_intent=model_configuration.add_intent,
                                            add_response_generator=model_configuration.add_response_generator,
                                            add_datasource=model_configuration.add_datasource,
                                            add_supports_screen=model_configuration.add_supports_screen,
                                            is_text_labels=is_text_2_text)
    print("Creating Validation Dataset")
    valid_dataset = RatingPredictionDataset(tokenizer=tokenizer, sessions=s_val, labels_list=l_val,
                                            num_labels=number_labels,
                                            number_turns_behaviour=model_configuration.number_turns_behaviour,
                                            scaling=scalling_method,
                                            number_turns_text=model_configuration.number_turns_text,
                                            system_separator_token=model_configuration.system_separator_token,
                                            user_separator_token=model_configuration.user_separator_token,
                                            max_system_words_per_turn=model_configuration.max_system_words_per_turn,
                                            replace_step_by_token=model_configuration.replace_step_by_token,
                                            replace_curiosity_by_token=model_configuration.replace_curiosity_by_token,
                                            add_intent=model_configuration.add_intent,
                                            add_response_generator=model_configuration.add_response_generator,
                                            add_datasource=model_configuration.add_datasource,
                                            add_supports_screen=model_configuration.add_supports_screen,
                                            is_text_labels=is_text_2_text)
    print("Creating Test Dataset")
    test_dataset = RatingPredictionDataset(tokenizer=tokenizer, sessions=s_test, labels_list=l_test,
                                           num_labels=number_labels,
                                           number_turns_behaviour=model_configuration.number_turns_behaviour,
                                           scaling=scalling_method,
                                           number_turns_text=model_configuration.number_turns_text,
                                           system_separator_token=model_configuration.system_separator_token,
                                           user_separator_token=model_configuration.user_separator_token,
                                           max_system_words_per_turn=model_configuration.max_system_words_per_turn,
                                           replace_step_by_token=model_configuration.replace_step_by_token,
                                           replace_curiosity_by_token=model_configuration.replace_curiosity_by_token,
                                           add_intent=model_configuration.add_intent,
                                           add_response_generator=model_configuration.add_response_generator,
                                           add_datasource=model_configuration.add_datasource,
                                           add_supports_screen=model_configuration.add_supports_screen,
                                           is_text_labels=is_text_2_text)

    if number_labels == 1:
        compute_metrics_function = compute_metrics_rating_prediction_regression
    else:
        if not is_text_2_text:
            compute_metrics_function = compute_metrics_rating_prediction_classification
        else:
            compute_metrics_function = compute_metrics_rating_prediction_seq2seq_classification

    # create the trainer object for training and validation
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=model_configuration.get_trainer_training_arguments(),  # get the training arguments
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        # metrics to compute
        compute_metrics=compute_metrics_function,
        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    )

    # actual training and validation procedure
    print("Started Training...")
    train_results = trainer.train()
    print("Training Results:")
    print(train_results)
    print()

    # actual test set evaluation procedure
    evaluator = Trainer(
        model=trainer.model,  # it loads the best model at the end of training
        args=model_configuration.get_trainer_testing_arguments(),
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_function,
        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    )

    # evaluate the best model from training on the validation set
    valid_results = evaluator.evaluate(valid_dataset, metric_key_prefix="valid")
    print("Validation Results:")
    print(valid_results)
    print()

    # evaluate the best model from training on the test set
    test_results = evaluator.evaluate(test_dataset)
    print("Test Results:")
    print(test_results)
    print()

    # join together validation and test results
    test_results.update(valid_results)

    checkpoint_name = extract_checkpoint_name(str(trainer.state.best_model_checkpoint))
    # saving the tokenizer
    tokenizer.save_pretrained(save_directory=os.path.join(model_configuration.output_dir, "tokenizer"))

    # print predictions
    print("Model Predictions on Test Set:")
    predictions_file_path = model_configuration.test_run_name + "_" + checkpoint_name + "_predictions.json"
    predictions_file_path = os.path.join(model_configuration.output_dir, predictions_file_path)
    _, all_predictions = get_predictions(evaluator=evaluator, tokenizer=tokenizer,
                                         number_labels=number_labels, test_dataset=test_dataset,
                                         labelling_method=labelling_method,
                                         write_predictions_to_file=predictions_file_path, print_to_console=False,
                                         is_text_2_text=is_text_2_text)
    print()

    # only count values for classification problem
    if model_configuration.labelling_method != labelling_method.Regression.value:
        # ignore the warning
        test_results.update({
            "labels_counter": {str(k): str(v) for k, v in Counter(l_test).items()},
            "preds_counter": {str(k): str(v) for k, v in Counter(all_predictions).items()}
        })

    # evaluate best model and write predictions to file
    config_metrics_results = write_model_results_to_file(model_configuration=model_configuration,
                                                         checkpoint=checkpoint_name, metrics=test_results)
    # create file with both config and metrics for easier access
    config_metrics_file_path = os.path.join(model_configuration.output_dir,
                                            model_configuration.test_run_name + checkpoint_name + "_all.json")
    with open(config_metrics_file_path, "w") as f_open:
        print(f"Model config and metrics written to file: {config_metrics_file_path}")
        json.dump(config_metrics_results, f_open, indent=2)
