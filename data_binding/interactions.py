import json
from datetime import datetime
from typing import Union, List, Dict

from tqdm import tqdm

from data_binding.session import Session


class AllInteractions:

    def __init__(self, file_paths: Union[str, List[str]], min_num_turns: int = None, only_rated: bool = None,
                 start_date: str = None, end_date: str = None, number_samples: int = None):

        self.file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        self.all_sessions = {}
        for file_path in self.file_paths:
            with open(file_path) as f_open:
                self.all_sessions.update(json.load(f_open))  # type: Dict

        self.sessions_dict_obj = {}
        for session_id, interaction in tqdm(self.all_sessions.items(), desc="Loading Sessions"):
            session_obj = Session(interaction)

            if session_obj and min_num_turns is not None and session_obj.get_number_dialogue_turns() < min_num_turns:
                session_obj = None
            if session_obj and only_rated and not session_obj.is_rated():
                session_obj = None
            if session_obj and start_date is not None:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
                if end_date is None:  # consider today
                    end_date_obj = datetime.today()
                else:
                    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

                if not (start_date_obj <= session_obj.start_time <= end_date_obj):
                    session_obj = None

            if session_obj is not None:
                self.sessions_dict_obj[session_id] = session_obj

            if number_samples is not None and len(self.sessions_dict_obj) >= number_samples:
                break

        print("Number of Sessions:", self.get_number_sessions())

    def count_number_interactions_per_user(self, user_id: str) -> int:
        count = 0
        for session in self.sessions_dict_obj.values():
            if session and user_id == session.user_id:
                count += 1
        return count

    def is_first_time_user(self, session_id: str) -> bool:
        current_session = self.get_by_session_id(session_id)
        for session in self.sessions_dict_obj.values():
            if current_session.session_id != session.session_id:
                if current_session.start_time > session.start_time > current_session.start_time:
                    return False
        return True

    def get_by_session_id(self, session_id: str):
        session = self.sessions_dict_obj.get(session_id, None)
        if session is None:
            raise ValueError("session id does not exist")
        else:
            return session

    def get_session_between_dates(self, start_date: str, end_date: str = None) -> List[Session]:
        sessions_list = []
        start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        if end_date is None:  # consider today
            end_date = datetime.today()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        for session in self.sessions_dict_obj.values():
            if start_date <= session.start_time <= end_date:
                sessions_list.append(session)
        return sessions_list

    def get_number_sessions(self):
        return len(self.sessions_dict_obj)

    def get_all_intents(self, convert_none_intent: bool = True):
        # just used to check all intents (can be used to add special tokens)
        intents = set()
        for session in self.sessions_dict_obj.values():
            for turn in session.turns_list_obj:
                if turn.intent:  # intent exists add to set
                    intents.add(turn.intent)
                elif convert_none_intent:  # intent is None however we want to consider it so convert to str and add
                    intents.add(str(turn.intent))
        return intents

    def get_all_response_generators(self, convert_none_response_generator: bool = True):
        # just used to check all response generators (can be used to add special tokens)
        response_generators = set()
        for session in self.sessions_dict_obj.values():
            for turn in session.turns_list_obj:
                if turn.candidate_responses_obj.active_responders:  # response_generator exists add to set
                    for rg in turn.candidate_responses_obj.active_responders:
                        response_generators.add(rg)
                elif convert_none_response_generator:
                    # response_generators is None however we want to consider it so convert to str and add
                    response_generators.add(str(turn.response_generator_chosen))
        return response_generators
