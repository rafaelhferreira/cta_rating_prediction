import json
import re
from datetime import datetime
from typing import Dict, Union, List, Tuple

from data_binding.enumerates import Emotions, DialogueResponses


class SentimentScore:

    def __init__(self, sentiment_score: Dict):
        self.sentiment_score = sentiment_score

        '''
        * Satisfaction – Measures the degree of positivity/negativity toward Alexa. Low satisfaction (-3) indicates 
        extreme frustration, whereas high satisfaction (+3) indicates delight. If your use-case involves 
        detecting/responding to Alexa defects, you'll likely want to use this dimension.
        * Activation – Measures the intensity of an emotion. High activation (+3) can arise from someone shouting from 
        excitement or anger, whereas low activation (-1) can arise if someone is subdued or tired sounding. 
        In practice, we rarely see activation scores < -1.
        * Valence – Measures the degree of positivity/negativity of an emotion in general. 
        Low valence (-3) can indicate extreme frustration (usually) or sadness, whereas high valence (+3) 
        indicates happiness. In practice, valence is highly correlated to satisfaction but differs in a few cases which 
        may be helpful in a conversational setting (e.g. if someone is generally sad irrespective of Alexa's response, 
        their valence would be low but their satisfaction would be neutral).
        '''

        self.valence = sentiment_score.get("valence", {})
        self.valence_score = None
        if self.valence:
            self.valence_score = float(self.valence.get("value"))

        self.activation = sentiment_score.get("activation", {})
        self.activation_score = None
        if self.activation:
            self.activation_score = float(self.activation.get("value"))

        self.satisfaction = sentiment_score.get("satisfaction", {})
        self.satisfaction_score = None
        if self.satisfaction:
            self.satisfaction_score = float(self.satisfaction.get("value"))

        self.emotion = self.calculate_emotion()

    def calculate_emotion(self) -> Union[str, None]:
        # heuristic defined (needs more study)
        emotion = Emotions.NONE
        if self.valence_score is not None and self.satisfaction_score is not None:  # we ignore activation here
            if self.valence_score >= 0.4 or self.satisfaction_score >= 0.4:
                emotion = Emotions.POSITIVE
            elif self.valence_score <= -0.4 or self.satisfaction_score <= -0.4:
                emotion = Emotions.NEGATIVE
            else:
                emotion = Emotions.NEUTRAL
        return emotion


class CandidateResponses:

    def __init__(self, candidate_response: Dict[str, str]):
        self.active_responders = list(candidate_response.keys())
        self.active_responses = list(candidate_response.values())
        self.candidate_response = candidate_response


class TokenAsr:

    def __init__(self, asr_token_dict: Dict):
        self.startOffsetInSeconds = int(asr_token_dict.get("startOffsetInMilliseconds", 0)) * 0.001  # type: float
        self.endOffsetInSeconds = int(asr_token_dict.get("endOffsetInMilliseconds", 0)) * 0.001  # type: float
        self.value = asr_token_dict.get("value")  # type: str
        self.confidence = float(asr_token_dict.get("confidence"))  # type: float


class AsrResponse:

    def __init__(self, asr_dict: List[Dict]):
        self.asr_options = asr_dict

        self.selected_asr_dict = self.asr_options[0] if self.asr_options else None
        self.tokens_list = []
        if self.selected_asr_dict:
            self.confidence_str = self.selected_asr_dict.get("confidence")  # type: str
            self.confidence = None
            if self.confidence_str is not None:
                self.confidence = float(self.confidence_str)  # type: float

            tokens_info = self.selected_asr_dict.get("tokens")  # type: Dict
            for token in tokens_info:
                self.tokens_list.append(TokenAsr(token))

        self.asr_min_value, self.asr_max_value, self.asr_avg_value = self._asr_min_max_avg_value()

    def _asr_min_max_avg_value(self) -> Tuple[float, float, float]:
        min_value = 1.0
        max_value = 0.0
        avg_value = 0.0
        for i, token in enumerate(self.tokens_list):
            if token.confidence < min_value:
                min_value = token.confidence
            if token.confidence > max_value:
                max_value = token.confidence
            avg_value += token.confidence
        avg_value = avg_value / len(self.tokens_list) if len(self.tokens_list) > 0 else 0
        return min_value, max_value, avg_value

    def response_latency(self) -> Union[None, float]:
        # we consider latency as the time it takes for the user to say the first token in the utterance
        latency = 0
        if self.tokens_list:
            latency = self.tokens_list[0].startOffsetInSeconds
        return latency

    def __str__(self):
        return str(self.asr_options)


class Turn:

    def __init__(self, session_dict: Dict):
        self.session_dict = session_dict
        self.session_id = session_dict.get("session_id", None)  # type: str

        self.creation_date_time_str = session_dict.get("creation_date_time", None)  # type: str

        if self.creation_date_time_str:
            self.creation_date_time = datetime.strptime(self.creation_date_time_str, '%Y-%m-%dT%H:%M:%S.%f')

        self.task_classifier_result = session_dict.get("task_classifier_result", None)  # type: str

        self.candidate_responses = session_dict.get("candidate_responses", {})  # type: Dict
        self.candidate_responses_obj = CandidateResponses(self.candidate_responses)

        self.intent = session_dict.get("intent")  # type str

        self.resume_task = session_dict.get("resume_task")  # type bool

        self.sentiment_score = session_dict.get("sentiment_score")  # type: Dict
        self.sentiment_score_obj = SentimentScore(self.sentiment_score)

        self.data_source = session_dict.get("data_source")  # type: str

        self.conversation_id = session_dict.get("conversation_id")  # type: str
        self.text = session_dict.get("text")  # type: str

        self.phase = session_dict.get("phase")  # type: str

        self.user_id = session_dict.get("user_id")  # type: str

        # 0 or 1
        self.input_offensive = bool(int(session_dict.get("input_offensive", 0)))  # type: bool

        self.response = session_dict.get("response", "")  # type: str

        self.cleaned_response = self.clean_response(self.response)

        self.system_latency = session_dict.get("system_latency")  # type: int

        self.current_step = session_dict.get("current_step")  # type: int
        self.current_method = session_dict.get("current_method")  # type: int

        self.asked_for_curiosity = any(self.remove_extra_spaces(i) in self.remove_extra_spaces(self.cleaned_response) for i in DialogueResponses.yes_no_confirmation_curiosity)
        self.said_curiosity = any(self.remove_extra_spaces(i) in self.remove_extra_spaces(self.cleaned_response) for i in DialogueResponses.fact_starters)

        self.is_step_text = self.is_step()

        self.response_generator_chosen = self.get_chosen_responder()

        self.asr = session_dict.get("asr")  # type: List
        self.asr_obj = None
        if self.asr:
            self.asr_obj = AsrResponse(self.asr)

    def get_chosen_responder(self) -> Union[str, None]:
        if len(self.candidate_responses_obj.active_responders) == 1:  # if there is only one it is the chosen one
            return self.candidate_responses_obj.active_responders[0]
        for rg_key, response in self.candidate_responses_obj.candidate_response.items():
            if self.response == response:  # if response is the same as candidate
                return rg_key
        return None

    def is_step(self) -> bool:
        # first step
        is_step = ("TASK_STEPS_RESPONDER" in self.candidate_responses_obj.active_responders or
                   'TASK_STEP_DETAIL_RESPONDER' in self.candidate_responses_obj.active_responders) \
                  and len(self.candidate_responses_obj.active_responders) == 1

        # all other steps start with this str or have the disclaimer
        if re.match(r"^(\s)*step \d\d*", self.cleaned_response, flags=re.IGNORECASE) or \
                DialogueResponses.safety_str in self.cleaned_response.lower():
            is_step = True

        return is_step

    def is_fallback(self) -> bool:
        for fallback_response in DialogueResponses.fallback_response + DialogueResponses.short_help_fallback_prefix:
            if self.response and (fallback_response.lower() in self.response.lower()
                                  or fallback_response.lower() in self.cleaned_response.lower()):
                return True
        return False

    def __str__(self):
        # just the str representation for printing purposes
        turn_str = {"response": self.response, "text": self.text, "intent": self.intent,
                    "phase": self.phase, "datasource": self.data_source}
        return json.dumps(turn_str, indent=2)

    @staticmethod
    def clean_response(response: str) -> str:
        try:
            cleaned_response = re.sub(r"<(?<=<).*?(?=>)>", " ", response).replace("&apos;", "'").replace(r'\xa0', r' ').replace("˚", "°")
            cleaned_response = re.sub(r"(.*[!.?])( *.$)", r"\1", cleaned_response)  # remove extra dot that is generally put at the end
            return cleaned_response.strip()
        except Exception as e:  # there is one that utterance that gives problems
            print(e)
            return response

    @staticmethod
    def remove_extra_spaces(response: str) -> str:
        response = response.strip()
        response = " ".join(response.split())
        return response
