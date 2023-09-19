import random
from enum import Enum
from typing import Union, List, Tuple, Dict, Any
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
from data_binding.session import Session
from data_binding.enumerates import DataSource
from models.session_features import SessionFeaturesVectors, ScalingMethod
import numpy as np
from collections import defaultdict
from collections import Counter

# seed for the dataset is always the same
DATASET_SEED = 42


class TokenizerSpecialTokens:
    SYSTEM = "[SYSTEM]"
    USER = "[USER]"

    SCREEN = "[SCREEN]"
    NO_SCREEN = "[NOSCREEN]"

    WIKIHOW = "[WIKIHOW]"
    RECIPE = "[RECIPE]"
    MULTISOURCE = "[MULTISOURCE]"

    STEP = "[STEP]"
    CURIOSITY = "[CURIOSITY]"

    @staticmethod
    def str_to_special_token_converter(text: str):
        # is used to add special tokens for example for intents and response generators
        # replaces adds []
        return str(f"[{str(text).replace('.', '')}]".upper())


class LabellingMethods(Enum):
    Binary_123_45 = "Binary_123_45"  # 1,2,3 -> 0, 4,5 -> 1
    Binary_12_345 = "Binary_12_345"  # 1,2 -> 0, 3,4,5 -> 1
    Triple_12_3_45 = "Triple_12_3_45"  # 1,2 -> 0, 3 -> 1, 4,5 -> 2
    Five_Way_12345 = "Five_Way_12345"  # 1 -> 0, 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4
    Regression = "Regression"


def convert_conversation_label(method: LabellingMethods, label_value: int) -> int:
    if method == LabellingMethods.Binary_12_345:
        return 0 if label_value < 3 else 1
    elif method == LabellingMethods.Binary_123_45:
        return 0 if label_value < 4 else 1
    elif method == LabellingMethods.Triple_12_3_45:
        if label_value < 3:
            return 0
        elif label_value < 4:
            return 1
        else:
            return 2
    elif method == LabellingMethods.Five_Way_12345:  # too specific
        return max(round(label_value)-1, 0)  # remove one to start the labels at zero
    elif method == LabellingMethods.Regression:
        return min(max(round(label_value), 0), 5)  # in regression we just round the value


def convert_number_to_str_repr(num: int) -> str:
    d = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    return d[num]  # this will return a zerp if it does not exist


def convert_str_to_number_repr(num_str: str) -> int:
    d = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    return d.get(num_str, -1)


def labeling_method_to_number_of_labels(method: LabellingMethods) -> int:
    if method == LabellingMethods.Binary_12_345.value or method == LabellingMethods.Binary_123_45.value:
        return 2
    elif method == LabellingMethods.Triple_12_3_45.value:
        return 3
    elif method == LabellingMethods.Five_Way_12345.value:
        return 5
    elif method == LabellingMethods.Regression.value:
        return 1  # in regression only one label


def session_to_text(session: Session, number_turns: Union[None, int],
                    system_separator_token: str,
                    user_separator_token: str,
                    max_system_words_per_turn: Union[int, None],
                    replace_step_by_token: Union[str, None],
                    replace_curiosity_by_token: Union[str, None],
                    add_intent: Union[bool, None],
                    add_response_generator: Union[bool, None],
                    add_datasource: Union[bool, None],
                    add_supports_screen: Union[bool, None]) -> str:

    session_text = ""
    if number_turns is None:  # consider all turns
        number_turns = len(session.turns_list_obj)

    for turn in session.turns_list_obj[max(0, len(session.turns_list_obj) - number_turns):]:  # get the last turns
        system_response = turn.cleaned_response
        # replace by special tokens if needed
        if replace_step_by_token and turn.is_step_text:
            system_response = replace_step_by_token
        elif replace_curiosity_by_token and turn.said_curiosity:
            system_response = replace_curiosity_by_token

        if max_system_words_per_turn is not None:  # cut the system response to a maximum size
            system_response = " ".join(system_response.split()[:max_system_words_per_turn]).strip()

        # concat system and user response with special separator token if needed
        intent = TokenizerSpecialTokens.str_to_special_token_converter(turn.intent) if add_intent else ""
        response_generator = TokenizerSpecialTokens.str_to_special_token_converter(turn.response_generator_chosen) if add_response_generator else ""

        session_text += f" {user_separator_token} {intent} {turn.text} {system_separator_token} {response_generator} {system_response} "

    data_source = ""
    if add_datasource and session.turns_list_obj[-1].data_source:
        data_source_str = session.turns_list_obj[-1].data_source
        if data_source_str == DataSource.RECIPE:
            data_source = TokenizerSpecialTokens.RECIPE
        elif data_source_str == DataSource.WIKIHOW:
            data_source = TokenizerSpecialTokens.WIKIHOW
        elif data_source_str == DataSource.MULTISOURCE:
            data_source = TokenizerSpecialTokens.MULTISOURCE

    has_screen = ""
    if add_supports_screen:
        if session.get_has_screen():
            has_screen = TokenizerSpecialTokens.SCREEN
        else:
            has_screen = TokenizerSpecialTokens.NO_SCREEN

    session_text = f"{has_screen} {data_source} {session_text}"

    return session_text.strip()


def split_dataset(sessions: List[Session], labelling_method: LabellingMethods) -> \
        Tuple[List[Session], List[Session], List[Session], List[int], List[int], List[int]]:

    labels = []
    # convert labels
    for session in sessions:
        labels.append(convert_conversation_label(method=labelling_method, label_value=session.rating))

    # set the seeds (for the creation of the dataset the seed is always the same)
    random.seed(DATASET_SEED)
    np.random.seed(DATASET_SEED)
    torch.manual_seed(DATASET_SEED)

    # 80% for training
    s_train, s_remain, l_train, l_remain = train_test_split(sessions, labels, test_size=0.2, shuffle=True,
                                                            random_state=DATASET_SEED, stratify=labels)

    # 10% eval and 10% for test
    s_val, s_test, l_val, l_test = train_test_split(s_remain, l_remain, test_size=0.5,
                                                    random_state=DATASET_SEED, stratify=l_remain)

    print("Dataset Statistics:")
    print("Train:", len(l_train), count_occurrences(l_train))
    print("Valid:", len(l_val), count_occurrences(l_val))
    print("Test:", len(l_test), count_occurrences(l_test))
    print()

    return s_train, s_val, s_test, l_train, l_val, l_test


def count_occurrences(labels: List) -> Dict[Any, int]:
    counts = {}
    for label in labels:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def special_truncation(sessions: List[Session],
                       tokenizer: PreTrainedTokenizer, batched_encoding: BatchEncoding, max_length: int,
                       add_datasource: Union[bool, None], add_supports_screen: Union[bool, None]):

    # it changes the input ids directly
    for i, input_ids in enumerate(batched_encoding.input_ids):
        if len(input_ids) >= max_length:  # only needs to update if it is greater or equal to max_length
            # these are the ones we check if they are correct
            token_1 = tokenizer.convert_ids_to_tokens(input_ids[1].item())
            token_2 = tokenizer.convert_ids_to_tokens(input_ids[2].item())

            # we must keep supports screen and datasource token
            if add_supports_screen and token_1 not in [TokenizerSpecialTokens.SCREEN,
                                                       TokenizerSpecialTokens.NO_SCREEN]:
                screen_token = TokenizerSpecialTokens.SCREEN if sessions[i].get_has_screen() else TokenizerSpecialTokens.NO_SCREEN
                input_ids[1] = tokenizer.vocab[screen_token]
            if add_datasource and token_2 not in [TokenizerSpecialTokens.WIKIHOW,
                                                  TokenizerSpecialTokens.RECIPE,
                                                  TokenizerSpecialTokens.MULTISOURCE]:
                datasource_token = None
                data_source = sessions[i].turns_list_obj[-1].data_source
                if data_source:
                    if data_source == DataSource.RECIPE:
                        datasource_token = TokenizerSpecialTokens.RECIPE
                    elif data_source == DataSource.WIKIHOW:
                        datasource_token = TokenizerSpecialTokens.WIKIHOW
                    elif data_source == DataSource.MULTISOURCE:
                        datasource_token = TokenizerSpecialTokens.MULTISOURCE
                if datasource_token:
                    input_ids[2] = tokenizer.vocab[datasource_token]


class RatingPredictionDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, sessions: List[Session], labels_list: List[int], num_labels: int,
                 number_turns_behaviour: int, scaling: ScalingMethod,
                 number_turns_text: int,
                 system_separator_token: str, user_separator_token: str, max_system_words_per_turn: Union[None, int],
                 replace_step_by_token: Union[str, None],
                 replace_curiosity_by_token: Union[str, None],
                 add_intent: Union[bool, None],
                 add_response_generator: Union[bool, None],
                 add_datasource: Union[bool, None],
                 add_supports_screen: Union[bool, None],
                 is_text_labels: bool = False):

        self.sessions = sessions
        self.max_length = tokenizer.model_max_length
        if self.max_length is None or self.max_length > 99999:
            self.max_length = 512

        self.all_sessions_text = []

        for session in sessions:  # convert a session to a text format to feed the tokenizer
            session_text = session_to_text(session=session, number_turns=number_turns_text,
                                           system_separator_token=system_separator_token,
                                           user_separator_token=user_separator_token,
                                           max_system_words_per_turn=max_system_words_per_turn,
                                           replace_step_by_token=replace_step_by_token,
                                           replace_curiosity_by_token=replace_curiosity_by_token, add_intent=add_intent,
                                           add_response_generator=add_response_generator,
                                           add_datasource=add_datasource, add_supports_screen=add_supports_screen)
            self.all_sessions_text.append(session_text)

        # tokenize the text
        self.inputs = tokenizer(self.all_sessions_text, add_special_tokens=True, truncation=True,
                                max_length=self.max_length, padding=True, return_tensors="pt")

        # this is to keep the datasource and supports screen tokens
        if tokenizer.truncation_side == "left":
            special_truncation(sessions=sessions, tokenizer=tokenizer, batched_encoding=self.inputs,
                               max_length=self.max_length,
                               add_datasource=add_datasource, add_supports_screen=add_supports_screen)

        # calculate behaviour features
        if number_turns_behaviour:
            behaviour_features = SessionFeaturesVectors(sessions_list=sessions, number_turns=number_turns_behaviour,
                                                        scaling=scaling)

            self.inputs["behaviour_features"] = torch.tensor(behaviour_features.sessions_behaviour_features)

        if not is_text_labels:
            if num_labels == 1:   # it is regression so use torch.float
                self.inputs["labels"] = torch.tensor(labels_list, dtype=torch.float)
            else:  # it is classification
                self.inputs["labels"] = torch.tensor(labels_list, dtype=torch.long)
        else:
            # is text labels so convert to input_ids
            self.inputs["labels"] = tokenizer([convert_number_to_str_repr(label) for label in labels_list],
                                              return_tensors="pt").input_ids

    def __len__(self):
        return len(self.inputs["labels"])

    def __getitem__(self, i: int):
        item = {key: val[i] for key, val in self.inputs.items()}
        return item


def calculate_session_list_stats(session_list: List[Session]) -> Dict:
    number_turns = []
    ratings = []
    # durations = []
    started_tasks = []
    supports_screen = []
    ratings_count = defaultdict(int)
    datasources = []
    number_steps_read = []
    all_feature_vectors = []
    for session in session_list:
        number_turns.append(len(session.turns_list_obj))
        ratings.append(session.rating)
        ratings_count[round(session.rating)] += 1  # we round the rating
        started_tasks.append(int(session.user_started_task()))
        supports_screen.append(int(session.get_has_screen()))
        datasources.append(session.datasource_by_turn(-1))
        number_steps_read.append(session.number_steps_read())

        feature_keys, feature_values = SessionFeaturesVectors.get_features_from_turn(session, number_turns[-1]-1)
        all_feature_vectors.append((feature_keys, feature_values))

    ratings_count = dict(sorted(ratings_count.items()))  # order by key

    datasources_names = []
    for d in datasources:
        datasources_names.append(DataSource.int_to_datasource[d])

    stats_dict = {
        "number_dialogues": len(session_list),
        "total_turns": sum(number_turns),
        "avg_number_turns": np.mean(number_turns),
        "std_number_turns": np.std(number_turns),
        "max_number_turns": max(number_turns),
        "min_number_turns": min(number_turns),
        "pearson_turns_vs_ratings": np.corrcoef(number_turns, ratings),
        # "pearson_durations_vs_ratings": pearsonr(durations, ratings),
        "pearson_started_task_vs_ratings": np.corrcoef(started_tasks, ratings),
        "pearson_supports_screen_vs_ratings": np.corrcoef(supports_screen, ratings),
        "pearson_datasource_vs_ratings": np.corrcoef(datasources, ratings),
        "pearson_number_steps_read_vs_ratings": np.corrcoef(number_steps_read, ratings),
        "datasources_counter": Counter(datasources_names),
        "ratings_count": ratings_count,
        "rating_count_percentage": {k: round(v / len(session_list) * 100, 1) for k, v in ratings_count.items()}
    }

    # calculate correlation with rating
    correlations = defaultdict(list)
    for session_features in all_feature_vectors:  # for each session feature
        for feature_name, feature_value in zip(session_features[0], session_features[1]):  # for each feature
            correlations[feature_name].append(feature_value)

    correlations_final = {}
    for k, v in correlations.items():
        correlations_final["pearson_" + k + "_vs_ratings"] = np.corrcoef(v, ratings)[0][1]
    # order by correlation
    correlations_final = dict(sorted(correlations_final.items(), key=lambda item: item[1], reverse=True))

    stats_dict.update(correlations_final)

    return stats_dict
