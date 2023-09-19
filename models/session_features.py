from statistics import mean
from typing import Union, List, Tuple

import numpy as np
from numpy import std
from data_binding.session import Session
from enum import Enum
from data_binding.enumerates import Intents, Phase
from tqdm import tqdm


# following https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-data
class ScalingMethod(Enum):
    StandardScalerMethod = "StandardScalerMethod"
    MinMaxScalerMethod = "MinMaxScalerMethod"
    MaxAbsScalerMethod = "MaxAbsScalerMethod"
    RobustScalerMethod = "RobustScalerMethod"


def apply_scaler(sessions_features: List[List[List[float]]], scaling_method: ScalingMethod):
    all_features = []
    for sessions_feature in sessions_features:
        for turn_feature in sessions_feature:
            for i, feature in enumerate(turn_feature):
                if len(all_features) <= i:  # handle first assigment
                    all_features.append([])
                all_features[i].append(feature)

    all_max_features, all_min_features, all_avg_features, all_std_features = [], [], [], []
    all_median_features, all_q75_features, all_q25_features = [], [], []
    for i, feature_list in enumerate(all_features):
        # this is very inneficient way to do this...
        all_max_features.append(max(feature_list))
        all_min_features.append(min(feature_list))
        all_avg_features.append(mean(feature_list))
        all_std_features.append(float(std(feature_list)))
        all_median_features.append(np.median(feature_list))
        q75, q25 = np.percentile(feature_list, [75, 25])
        all_q75_features.append(q75)
        all_q25_features.append(q25)

    all_sessions_scaled_features = []
    for sessions_feature in sessions_features:
        scaled_session_features = []
        for turn_feature in sessions_feature:
            scaled_turn_features = []
            for i in range(len(turn_feature)):
                if scaling_method == ScalingMethod.StandardScalerMethod:
                    # StandardScaler
                    if all_std_features[i] != 0:
                        scaled_value = (turn_feature[i] - all_avg_features[i]) / all_std_features[i]
                    else:
                        scaled_value = 0
                elif scaling_method == ScalingMethod.MinMaxScalerMethod:
                    # MinMaxScaler
                    if (all_max_features[i] - all_min_features[i]) != 0:
                        x_std = (turn_feature[i] - all_min_features[i]) / (all_max_features[i] - all_min_features[i])
                        max_range = 1
                        min_range = 0
                        scaled_value = x_std * (max_range - min_range) + min_range
                    else:
                        scaled_value = 0
                elif scaling_method == ScalingMethod.MaxAbsScalerMethod:
                    # MaxAbsScaler
                    if all_max_features[i] != 0:
                        scaled_value = turn_feature[i] / all_max_features[i]
                    else:
                        scaled_value = 0
                elif scaling_method == ScalingMethod.RobustScalerMethod:
                    # RobustScaler
                    if (all_q75_features[i] - all_q25_features[i]) != 0:
                        scaled_value = (turn_feature[i] - all_median_features[i]) / (all_q75_features[i] - all_q25_features[i])
                    else:
                        scaled_value = 0
                else:
                    raise ValueError(f"Scaling method {scaling_method} is not a valid value.")

                scaled_turn_features.append(scaled_value)
            scaled_session_features.append(scaled_turn_features)
        all_sessions_scaled_features.append(scaled_session_features)

    return all_sessions_scaled_features


class SessionFeaturesVectors:

    def __init__(self, sessions_list: List[Session], number_turns: int, scaling: Union[None, ScalingMethod]):

        self.sessions_list = sessions_list
        self.number_turns = number_turns
        self.scaling = scaling

        self.sessions_behaviour_features = []
        self.session_features_names = None
        for session in tqdm(sessions_list, desc="Creating Session Features"):
            self.session_features_names, session_features = self.get_session_behaviour_features(session=session,
                                                                                                number_turns=number_turns)
            self.sessions_behaviour_features.append(session_features)

        if scaling:
            self.sessions_behaviour_features = apply_scaler(sessions_features=self.sessions_behaviour_features,
                                                            scaling_method=scaling)

    def get_session_behaviour_features(self, session: Session, number_turns: int) -> \
            Tuple[List[List[str]], List[List[float]]]:

        turns_features_list = []
        turn_features_names_list = []
        for i in range(len(session.turns_list_obj)-number_turns, len(session.turns_list_obj)):
            turn_features_names, turn_features_values = self.get_features_from_turn(session=session, turn_index=i)
            turns_features_list.append(turn_features_values)
            turn_features_names_list.append(turn_features_names)
        return turn_features_names_list, turns_features_list

    @staticmethod
    def get_features_from_turn(session: Session, turn_index: int) -> Tuple[List[str], List[float]]:
        features = {
            # behaviour features
            "SessionDuration": session.get_total_session_time(turn_index),
            "Turns": turn_index+1,
            "UtterancePos": session.get_number_positive(turn_index),
            "UtteranceNeg": session.get_number_negative(turn_index),
            "AvgUtterancePos": session.get_number_positive(turn_index) / (turn_index+1),
            "AvgUtteranceNeg": session.get_number_negative(turn_index) / (turn_index+1),
            "Offensive": session.number_turns_offensive(turn_index),  # maybe avg
            "Sensitive": session.number_turns_sensitive(turn_index),  # maybe avg
            "Screens": session.screens_visited_with_repetitions(turn_index),
            "Searches": session.number_searches(turn_index),
            "RepeatedSearches": session.number_of_repeated_searches(turn_index),
            "ResultPages": session.number_result_pages(turn_index),
            "StartedTask": int(session.user_started_task(turn_index)),
            "FinishedTask": int(session.user_finished_task(turn_index)),
            "FallbackExceptions": session.number_of_fallback_exceptions(turn_index),
            "UserWordOverlap": session.token_overlaps_user(turn_index),
            "SystemWordOverlap": session.token_overlaps_system(turn_index),
            "UserSystemWordOverlap": session.token_overlap_user_system(turn_index),
            "AvgUserWordOverlap": session.avg_token_overlaps_user(turn_index),
            "AvgSystemWordOverlap": session.avg_token_overlaps_system(turn_index),
            "AvgUserSystemWordOverlap": session.avg_token_overlaps_user_system(turn_index),
            "WordsUser": session.user_response_number_words(turn_index),  # only for turn not like in paper
            "WordsSystem": session.system_response_number_words(turn_index),  # only for turn not like in paper
            "AvgWordsUser": session.total_number_words(turn_index, is_user=True) / (turn_index+1),
            "AvgWordsSystem": session.total_number_words(turn_index, is_user=False) / (turn_index+1),
            "UniqueUserWords": session.unique_tokens(turn_index, is_user=True),
            "UniqueSystemWords": session.unique_tokens(turn_index, is_user=False),

            # system features
            "SystemLatency": session.get_system_latency(turn_index),
            "AvgSystemLatency": session.get_system_avg_latency(turn_index),
            "MaxSystemLatency": session.get_system_max_latency(turn_index),
            "TurnDuration": session.get_turn_duration(turn_index),
            "AvgTurnDuration": session.get_avg_duration_per_turn(turn_index),
            "MaxTurnDuration": session.get_max_duration_per_turn(turn_index),
            "UserLatency": session.user_latency(turn_index),
            "MaxUserLatency": session.user_latency_max(turn_index),
            "AvgUserLatency": session.user_latency_avg(turn_index),
            "ASR_Min": session.min_token_confidence(turn_index),
            "ASR_Max": session.max_token_confidence(turn_index),
            "ASR_Avg": session.avg_token_confidence(turn_index),

            # topic preference features
            "StepsRead": session.number_steps_read(turn_index),
            "RepeatedUserUtterance": session.consecutive_user_response(turn_index),
            "RepeatedSystemUtterance": session.consecutive_bot_response(turn_index),
            "Resumed": int(session.session_with_resume(turn_index)),
            "HasScreen": int(session.get_has_screen()),
            "Domain": session.datasource_by_turn(turn_index),
        }

        # other methods
        curiosities_accepted, curiosities_denied = session.curiosities_accepted_denied(turn_index)
        features["CuriositiesAccepted"] = curiosities_accepted
        features["CuriositiesDenied"] = curiosities_denied
        features["CuriositiesSaid"] = session.number_curiosities_said(turn_index)

        # turns per screen
        for screen_name, count in session.number_turns_per_all_screens(turn_index).items():
            pretty_name = Phase.convert_to_pretty_name(screen_name)
            features[pretty_name] = count

        # consider the counts of the intents
        important_intents = [Intents.CancelIntent, Intents.RepeatIntent, Intents.YesIntent,
                             Intents.NoIntent, Intents.IngredientsConfirmationIntent,
                             Intents.NoneOfTheseIntent, Intents.StartCookingIntent,
                             Intents.StartStepsIntent, Intents.TerminateCurrentTaskIntent,
                             Intents.IdentifyProcessIntent, Intents.FallbackIntent,
                             Intents.MoreDetailIntent, Intents.NextStepIntent, Intents.NextIntent,
                             Intents.HelpIntent]
        for intent in important_intents:
            pretty_name = Intents.convert_to_pretty_name(intent)
            features[pretty_name] = session.number_turns_with_intent(intents=[intent], turn_index=turn_index)

        return list(features.keys()), list(features.values())
