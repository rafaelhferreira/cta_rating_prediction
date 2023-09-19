class Intents:
    CancelIntent = 'CancelIntent'
    RepeatIntent = 'RepeatIntent'
    NoIntent = 'NoIntent'
    YesIntent = 'YesIntent'
    IngredientsConfirmationIntent = 'IngredientsConfirmationIntent'
    NoneOfTheseIntent = 'NoneOfTheseIntent'
    PreviousStepIntent = 'PreviousStepIntent'
    PreviousIntent = 'PreviousIntent'
    StartCookingIntent = 'StartCookingIntent'
    StartStepsIntent = 'StartStepsIntent'
    TerminateCurrentTaskIntent = 'TerminateCurrentTaskIntent'
    QuestionIntent = 'QuestionIntent'
    IdentifyProcessIntent = 'IdentifyProcessIntent'  # search intent
    MoreDetailIntent = 'MoreDetailIntent'
    NextIntent = 'NextIntent'
    NextStepIntent = 'NextStepIntent'
    FallbackIntent = 'FallbackIntent'
    HelpIntent = 'HelpIntent'
    StopIntent = 'StopIntent'

    @staticmethod
    def is_back_transition_intent(intent: str):
        return intent in {Intents.PreviousIntent, Intents.PreviousStepIntent, Intents.CancelIntent}

    @staticmethod
    def convert_to_pretty_name(intent: str):
        pretty_names = {
            "IdentifyProcessIntent": "Search",
            "NoneOfTheseIntent": "None of These",
            "CancelIntent": "Cancel",
            "YesIntent": "Yes",
            "NoIntent": "No",
            "IngredientsConfirmationIntent": "Ingredients",
            "StartCookingIntent": "Start Cooking",
            "StartStepsIntent": "Start Steps",
            "NextIntent": "Next",
            "NextStepIntent": "Next Step",
            "MoreDetailIntent": "More Detail",
            "TerminateCurrentTaskIntent": "Terminate Task",
            "HelpIntent": "Help",
            "RepeatIntent": "Repeat",
            "FallbackIntent": "Fallback"
        }

        intent = pretty_names.get(intent, intent)
        intent = intent.replace("Intent", "")

        intent = "Intent: " + intent

        return intent


class DataSource:
    WIKIHOW = 'wikihow'
    RECIPE = 'recipe'
    MULTISOURCE = 'multi_source'

    datasource_to_int = {
        RECIPE: 0,
        WIKIHOW: 1,
        None: 2,
        MULTISOURCE: 3
    }

    int_to_datasource = {
        0: RECIPE,
        1: WIKIHOW,
        2: None,
        3: MULTISOURCE
    }


class Phase:
    GREETING = 'greeting'
    PROCESS_GROUNDING = 'process_grounding'
    TASK_OVERVIEW = 'task_overview'
    INGREDIENT_LIST = 'ingredient_list'
    STEPS = 'steps'
    STEP_DETAIL = 'step_detail'
    STEPS_CONCLUDED = 'steps_concluded'

    @staticmethod
    def is_in_task(current_phase: str) -> bool:
        return current_phase in {Phase.TASK_OVERVIEW, Phase.INGREDIENT_LIST, Phase.STEPS,
                                 Phase.STEP_DETAIL, Phase.STEPS_CONCLUDED}

    @staticmethod
    def get_all_phase_names():
        return [Phase.GREETING, Phase.PROCESS_GROUNDING, Phase.TASK_OVERVIEW,
                Phase.INGREDIENT_LIST, Phase.STEPS, Phase.STEP_DETAIL, Phase.STEPS_CONCLUDED]

    @staticmethod
    def convert_to_pretty_name(current_phase: str):
        pretty_names = {
            "greeting": "Greeting",
            "process_grounding": "Search",
            "task_overview": "TaskOverview",
            "ingredient_list": "Ingredients",
            "steps": "Steps",
            "step_detail": "Step\'s Detail",
            "steps_concluded": "Conclusion",
        }

        current_phase = pretty_names.get(current_phase, None)
        if current_phase:
            current_phase = "Phase: " + current_phase

        return current_phase


class DialogueResponses:

    fallback_response = [
        "I'm not sure I got that. To find out what I can do, just ask me for \'help\'.",
        "I'm not sure I got that. But you can ask me for \'help\'.",
        "I am not sure I got that. To find out what I can do, just ask me for \'help\'.",
        "I am not sure I got that. But you can ask me for \'help\'.",
        "Wellll. If you're not sure what to say next, ask me for \'help\'.",
        "Hm, that's a tough-one. But you can ask me for \'help\'.",
        "that's a tough-one.",
        "Sorry, I don't know that",
        "I'm still learning",
        "Hm, I heard",
    ]

    short_help_fallback_prefix = [
        "I think I missed that.",
        "I think I misheard that.",
    ]

    answer_to_question_not_found = [
        "Hm, I'm not sure about that.",
        "Hm, that's a tough-one. I don't know the answer to that question.",
        "I couldn't find an answer to that. I'm still learning.",
    ]

    yes_no_confirmation_curiosity = [
        "fun fact about this",
        "fun fact about this",
        "astonish you with a fun fact",
    ]

    fact_starters = [
        # emotions removed due to response and response cleaned differences
        "Did you know that:",
        "Curiosity time:",
        "It's interesting to know that:",
        "How crazy is it that:",
        "Alert! Alert!",
        "Watch out!",
        "Do you believe that",
        "here comes a curiosity just for you",
        "a fun fact just for you",
        "i also love fun facts! here it goes",
        "for a curious mind, a curious fact!",
        "a fun fact a day, keeps you happy all the way!",
    ]

    # when the user searches for the same thing twice
    repeated_searchs_str_list = [
        "We are already searching for that. If the results weren't what you wanted, try checking a different d.i.y. task.",
        "We are already searching for that. If the results weren't what you wanted, try saying 'more options', or check a different d.i.y. task.",
        "We are already searching for that. If the results weren't what you wanted, you can try our delicious weekly suggestions.",
        "We are already searching for that. If the results weren't what you wanted, try saying 'more options', or you can try our delicious weekly suggestions.",
        "We are already searching for that",
    ]

    safety_str = "Please be careful when using any tools or equipment"


class Emotions:
    NONE = None
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
