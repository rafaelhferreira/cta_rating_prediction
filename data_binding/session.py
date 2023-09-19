import json
from datetime import datetime
from statistics import mean
from typing import Dict, List, Union, Tuple

from data_binding.enumerates import Phase, DialogueResponses, Intents, Emotions, DataSource
from data_binding.turn import Turn


class Session:

    def __init__(self, session_dict: Dict):
        self.session_dict = session_dict
        self.session_id = session_dict.get("session_id")  # type: str
        self.user_id = session_dict.get("user_id")  # type: str
        self.conversation_id = session_dict.get("conversation_id")  # type: str
        self.start_time_str = session_dict.get("start_time")  # type: str
        self.start_time = None
        if self.start_time_str:
            self.start_time = datetime.strptime(self.start_time_str, '%Y-%m-%dT%H:%M:%S.%f')

        self.rating_timestamp_str = session_dict.get("rating_timestamp")  # type: str
        self.rating_timestamp = None
        if self.rating_timestamp_str:
            try:
                self.rating_timestamp = datetime.strptime(self.rating_timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(e, f"When converting {self.rating_timestamp_str} to datetime object")

        self.duration_str = session_dict.get("duration")  # type: str
        self.duration = None
        if self.duration_str:
            try:
                self.duration = float("".join(self.duration_str.split()[-1]))
            except ValueError as e:
                print(e, f"When converting {self.duration_str} to float")

        self.rating_str = session_dict.get("rating")  # type: str
        self.rating = None
        if self.rating_str:
            self.rating = float(self.rating_str)

        self.turns_list = session_dict.get("turns", [])  # type: List[Dict]
        self.turns_list_obj = []
        for session in self.turns_list:
            self.turns_list_obj.append(Turn(session))

        # set the datasources correctly (because of bug datasource is removed after user chooses a task)
        self.__fill_datasource()

        self.has_screen = session_dict.get("has_screen", False)  # type: bool

    def get_number_dialogue_turns(self):
        return len(self.turns_list_obj)

    def get_avg_duration_per_turn(self, turn_index: int = None) -> float:
        avg_time = []
        number_turns = self.get_number_dialogue_turns()
        if turn_index is None:
            turn_index = -1
        else:
            turn_index = min(len(self.turns_list_obj) - 1, turn_index + 1)
        if number_turns > 1:
            for i, turn in enumerate(self.turns_list_obj[:turn_index]):
                avg_time.append((self.turns_list_obj[i + 1].creation_date_time - turn.creation_date_time).seconds)
        return mean(avg_time)

    def get_max_duration_per_turn(self, turn_index: int = None) -> float:
        max_time = 0
        number_turns = self.get_number_dialogue_turns()
        if turn_index is None:
            turn_index = -1
        else:
            turn_index = min(len(self.turns_list_obj) - 1, turn_index + 1)
        if number_turns > 1:
            for i, turn in enumerate(self.turns_list_obj[:turn_index]):
                turn_duration = self.get_turn_duration(i)
                if turn_duration > max_time:
                    max_time = turn_duration
        return max_time

    def get_turn_duration(self, turn_number: int) -> Union[float, None]:
        if turn_number < self.get_number_dialogue_turns() - 1:  # for the last one we cannot calculate this
            return (self.turns_list_obj[turn_number + 1].creation_date_time -
                    self.turns_list_obj[turn_number].creation_date_time).seconds
        else:
            return 0

    def get_system_max_latency(self, turn_index: int = None) -> float:
        # not really correct
        max_latency = -1
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            current_latency = turn.system_latency
            if current_latency > max_latency:
                max_latency = current_latency
        return max_latency if max_latency > 0 else None

    def get_system_avg_latency(self, turn_index: int = None):
        # not really correct
        avg_latency = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            avg_latency += turn.system_latency
        return avg_latency / self.get_number_dialogue_turns()

    def get_turns_in_screen(self, screens: List[str]) -> List[Turn]:
        turns_in_screen = []
        for turn in self.turns_list_obj:
            if turn.phase in screens:
                turns_in_screen.append(turn)
        return turns_in_screen

    def get_system_latency(self, turn_number: int):
        # not really correct
        return self.turns_list_obj[turn_number].system_latency

    def get_total_session_time(self, turn_index: int = -1, ignore_limit: bool = False):
        if self.get_number_dialogue_turns() > 1:
            if ignore_limit:  # count from beginning to end
                return (self.turns_list_obj[turn_index].creation_date_time - self.turns_list_obj[0].creation_date_time).seconds
            else:
                # we use min to limit the maximum duration of a session to 30 min
                # if it takes more than that 30 min it is probably a resume
                return min((self.turns_list_obj[turn_index].creation_date_time - self.turns_list_obj[0].creation_date_time).seconds, 1800)
        return 0  # only one turn return 0

    def user_started_task(self, turn_index: int = None):
        return self.has_visited_screen([Phase.STEPS, Phase.STEP_DETAIL, Phase.STEPS_CONCLUDED], turn_index)

    def user_finished_task(self, turn_index: int = None):
        return self.has_visited_screen(Phase.STEPS_CONCLUDED, turn_index)

    def has_visited_screen(self, screens: List[str], turn_index: int = None):
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for i, turn in enumerate(self.turns_list_obj[:turn_index]):
            if turn.phase and turn.phase in screens:
                return True
        return False

    def get_has_screen(self) -> bool:
        return self.has_screen

    def number_of_dialogue_exceptions(self) -> int:
        # when it goes to the non-contextual fallback
        return self.count_number_of_times_str(DialogueResponses.fallback_response)

    def number_of_fallback_exceptions(self, turn_index: int = None) -> int:
        return self.count_number_of_times_str(list_to_consider=DialogueResponses.fallback_response +
                                                               DialogueResponses.short_help_fallback_prefix,
                                              turn_index=turn_index)

    def number_answers_without_response(self, turn_index: int = None) -> int:
        return self.count_number_of_times_str(list_to_consider=DialogueResponses.answer_to_question_not_found,
                                              intents=[Intents.QuestionIntent, Intents.IdentifyProcessIntent],
                                              turn_index=turn_index)

    def count_number_of_times_str(self, list_to_consider: List[str], intents: List[str] = None,
                                  turn_index: int = None) -> int:
        # count everytime it contains one of these strings
        count = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            for response in list_to_consider:
                if turn.response and (response.lower() in turn.response.lower() or response.lower() in turn.cleaned_response.lower()):
                    if not intents:
                        count += 1
                    elif intents and turn.intent in intents:  # if we want to consider only some intents
                        count += 1
        return count

    def number_distinct_intents(self) -> int:
        intents = set()
        for turn in self.turns_list_obj:
            intents.add(turn.intent)
        return len(intents)

    def number_turns_with_intent(self, intents: List[str], turn_index: int = None):
        count = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.intent in intents:
                count += 1
        return count

    def number_repeats(self):
        return self.number_turns_with_intent([Intents.RepeatIntent])

    def number_turns_offensive(self, turn_index: int = None):
        # as classifier by amazon's offensive classifier
        counter = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.input_offensive:
                counter += 1
        return counter

    def number_turns_sensitive(self, turn_index: int = None) -> int:
        # as classifier by amazon's offensive classifier
        counter = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if "SENSITIVE_RESPONDER" in turn.candidate_responses_obj.active_responders:
                counter += 1
        return counter

    def number_steps_read(self, turn_index: int = None) -> int:
        counter = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.is_step_text:
                counter += 1
        return counter

    def number_curiosities_said(self, turn_index: int = None):
        counter = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            candidate_responders = turn.candidate_responses_obj.active_responders
            if ("CURIOSITY_RESPONDER" in candidate_responders and len(candidate_responders) == 1) or turn.said_curiosity:
                counter += 1
        return counter

    def curiosities_accepted_denied(self, turn_index: int = None) -> Tuple[int, int]:
        number_accepted, number_denied = 0, 0
        if turn_index is None:
            turn_index = -1
        else:
            turn_index = min(len(self.turns_list_obj) - 1, turn_index + 1)
        if self.get_number_dialogue_turns() > 1:
            for i, turn in enumerate(self.turns_list_obj[:turn_index]):
                if turn.asked_for_curiosity:  # asked for curiosity
                    if self.turns_list_obj[i + 1].intent == Intents.YesIntent:
                        # asked for curiosity and user said Yes
                        number_accepted += 1
                    elif self.turns_list_obj[i + 1].intent == Intents.NoIntent:
                        # asked for curiosity and user said No
                        number_denied += 1
        return number_accepted, number_denied

    def get_last_step(self) -> Union[int, None]:
        return self.turns_list_obj[-1].current_step  # if it has returns a number else None

    def consecutive_user_response(self, turn_index: int = None) -> int:
        count = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        current_response = self.turns_list_obj[0].text
        for turn in self.turns_list_obj[1:turn_index]:
            if current_response == turn.text:
                count += 1
            current_response = turn.text
        return count

    def consecutive_user_intent(self) -> int:
        count = 0
        current_intent = self.turns_list_obj[0].intent
        for turn in self.turns_list_obj[1:]:
            if current_intent == turn.intent:
                count += 1
            current_intent = turn.intent
        return count

    def consecutive_bot_response(self, turn_index: int = None) -> int:
        count = 0
        current_bot_response = self.turns_list_obj[0].response
        for turn in self.turns_list_obj[1:turn_index]:
            if current_bot_response == turn.response:
                count += 1
            current_bot_response = turn.response
        return count

    def number_back_transitions(self):
        count = 0
        current_screen = self.turns_list_obj[0].phase
        for i, turn in enumerate(self.turns_list_obj[1:]):
            # it is a 'previous' and it changed screen indicating a transition
            if Intents.is_back_transition_intent(turn.intent) and \
                    current_screen != current_screen and turn.phase != current_screen:
                count += 1
            current_screen = turn.phase

    def number_turns_per_screen(self, phase: str) -> int:
        count = 0
        for turn in self.turns_list_obj:
            if phase == turn.phase:
                count += 1
        return count

    def number_turns_per_all_screens(self, turn_index: int = None) -> Dict[str, int]:
        count_per_screen = {}
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for i in Phase.get_all_phase_names():  # do this to have all screens in the same order
            count_per_screen[i] = 0
        for turn in self.turns_list_obj[:turn_index]:
            if turn.phase:
                if turn.phase not in count_per_screen:
                    count_per_screen[turn.phase] = 0
                count_per_screen[turn.phase] += 1
        return count_per_screen

    def number_searches(self, turn_index: int = None):
        count = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            # the amount of times task_classifier is called gives the number of searches
            # not counting No intents since these already account for the search
            if (turn.task_classifier_result is not None and turn.intent not in [Intents.NoIntent]) or \
                    "USER_EVENT_IDENTIFY_PROCESS_RESPONDER" in turn.candidate_responses_obj.active_responders:
                count += 1
        return count

    def number_of_repeated_searches(self, turn_index: int = None):
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        return self.count_number_of_times_str(DialogueResponses.repeated_searchs_str_list, turn_index=turn_index)

    def number_curiosities_asked(self, turn_index: int = None):
        count = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.asked_for_curiosity:
                count += 1
        return count

    def number_result_pages(self, turn_index: int = None):
        count = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.phase == Phase.PROCESS_GROUNDING \
                    and turn.intent in [Intents.MoreDetailIntent, Intents.NextStepIntent, Intents.NextIntent]:
                count += 1
        return count

    def is_rated(self) -> bool:
        return bool(self.rating)

    def session_with_resume(self, turn_index: int = None) -> bool:
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.resume_task:
                return True
        return False

    def get_number_positive(self, turn_index: int = None) -> int:
        count = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.sentiment_score_obj.emotion == Emotions.POSITIVE:
                count += 1
        return count

    def get_number_negative(self, turn_index: int = None) -> int:
        count = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.sentiment_score_obj.emotion == Emotions.NEGATIVE:
                count += 1
        return count

    def screens_visited(self) -> int:
        # visited screens without repetitions
        visited_screens = set()
        for turn in self.turns_list_obj:
            visited_screens.add(turn.phase)
        return len(visited_screens)

    def screens_visited_with_repetitions(self, turn_index: int = None) -> int:
        # visited screens counting repeats
        visited_screens = []
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if not visited_screens or visited_screens[-1] != turn.phase:
                visited_screens.append(turn.phase)
        return len(visited_screens)

    def token_overlaps_user(self, turn_number: int) -> float:
        if turn_number >= 1:
            current_turn_text = self.turns_list_obj[turn_number].text
            previous_turn_text = self.turns_list_obj[turn_number - 1].text
            return self.calculate_token_overlap(current_turn_text, previous_turn_text)
        return 0

    def token_overlaps_system(self, turn_number: int) -> float:
        if turn_number >= 1:
            current_turn_text = self.turns_list_obj[turn_number].cleaned_response
            previous_turn_text = self.turns_list_obj[turn_number - 1].cleaned_response
            return self.calculate_token_overlap(current_turn_text, previous_turn_text)
        return 0

    def token_overlap_user_system(self, turn_number: int) -> float:
        if turn_number >= 1:
            user_response = self.turns_list_obj[turn_number].text
            system_response = self.turns_list_obj[turn_number - 1].cleaned_response
            # check if user is mimicking system utterance in the previous turn
            return self.calculate_token_overlap(user_response, system_response)
        return 0

    def avg_token_overlaps_user(self, turn_index: int) -> float:
        avg_overlap = []
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for i in range(turn_index):
            avg_overlap.append(self.token_overlaps_user(i))
        return mean(avg_overlap)

    def avg_token_overlaps_system(self, turn_index: int) -> float:
        avg_overlap = []
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for i in range(turn_index):
            avg_overlap.append(self.token_overlaps_system(i))
        return mean(avg_overlap)

    def avg_token_overlaps_user_system(self, turn_index: int) -> float:
        avg_overlap = []
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for i in range(turn_index):
            avg_overlap.append(self.token_overlap_user_system(i))
        return mean(avg_overlap)

    def user_response_number_words(self, turn_number: int):
        user_response = self.turns_list_obj[turn_number].text
        if user_response:
            return len(user_response.split())
        else:
            return 0

    def system_response_number_words(self, turn_number: int):
        system_response = self.turns_list_obj[turn_number].cleaned_response
        if system_response:
            return len(system_response.split())
        else:
            return 0

    def total_number_words(self, turn_index: int, is_user: bool) -> int:
        number_words = 0
        for turn in self.turns_list_obj[:turn_index]:
            if is_user and turn.text:
                number_words += len(turn.text.split())
            elif turn.cleaned_response:  # is system
                number_words += len(turn.cleaned_response.split())
        return number_words

    @staticmethod
    def calculate_token_overlap(current_turn: str, previous_turn: str) -> Union[float, None]:
        if current_turn and previous_turn:
            current_tokens = set(current_turn.split())
            previous_tokens = set(previous_turn.split())
            return len(current_tokens.intersection(previous_tokens)) / len(current_tokens)
        else:
            return 0

    def unique_tokens(self, turn_index: int, is_user: bool):
        user_response = self.turns_list_obj[turn_index].text
        system_response = self.turns_list_obj[turn_index - 1].cleaned_response
        # check unique tokens between current user_response and previous system
        if is_user:
            return self.calculate_unique_tokens(user_response, system_response)
        else:
            return self.calculate_unique_tokens(system_response, user_response)

    @staticmethod
    def calculate_unique_tokens(first, second) -> Union[int, None]:
        if first and second:
            current_tokens = set(first.split())
            previous_tokens = set(second.split())
            return len(current_tokens.difference(previous_tokens))
        else:
            return 0

    def min_token_confidence(self, turn_index: int):
        asr_obj = self.turns_list_obj[turn_index].asr_obj
        if asr_obj:
            return asr_obj.asr_min_value
        else:
            return 0

    def max_token_confidence(self, turn_index: int):
        asr_obj = self.turns_list_obj[turn_index].asr_obj
        if asr_obj:
            return asr_obj.asr_max_value
        else:
            return 0

    def avg_token_confidence(self, turn_index: int):
        asr_obj = self.turns_list_obj[turn_index].asr_obj
        if asr_obj:
            return asr_obj.asr_avg_value
        else:
            return 0

    def user_latency(self, turn_index: int) -> float:
        # time to say the first token in seconds
        asr_obj = self.turns_list_obj[turn_index].asr_obj
        if asr_obj:
            return asr_obj.response_latency()
        else:
            return 0

    def __fill_datasource(self):
        # because datasource is deleted when enters a task we will fill it here
        for i, turn in enumerate(self.turns_list_obj[:-1]):
            if turn.data_source and not self.turns_list_obj[i+1].data_source:
                # if current turn has datasource and the following one does not we put the same as the one before
                self.turns_list_obj[i+1].data_source = turn.data_source

    def datasource_by_turn(self, turn_index: int) -> int:
        # returns the conversion of the datasource to int
        return DataSource.datasource_to_int[self.turns_list_obj[turn_index].data_source]

    def user_latency_avg(self, turn_index: int = None) -> float:
        # time to say the first token in seconds
        avg_latency = []
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.asr_obj:
                avg_latency.append(turn.asr_obj.response_latency())
        return mean(avg_latency)

    def user_latency_max(self, turn_index: int = None) -> float:
        # time to say the first token in seconds
        max_latency = 0
        if turn_index is not None:
            turn_index += 1  # sum one to account for the current turn
        for turn in self.turns_list_obj[:turn_index]:
            if turn.asr_obj and max_latency < turn.asr_obj.response_latency():
                max_latency = turn.asr_obj.response_latency()
        return max_latency

    def session_to_simple_dict(self) -> List[Dict]:
        session = []
        for turn in self.turns_list_obj:
            session.append({"response": turn.response, "text": turn.text, "intent": turn.intent,
                            "phase": turn.phase, "datasource": turn.data_source})
        return session

    def __str__(self):
        # just the str representation for printing purposes
        return json.dumps(self.session_to_simple_dict(), indent=2)
