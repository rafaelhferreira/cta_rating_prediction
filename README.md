# Rating Prediction in Conversational Task Assistants with Behavioral and Conversational-Flow Features

This is the repository for the paper Rating Prediction in Conversational Task Assistants with
Behavioral and Conversational-Flow Features published at SIGIR 2023 [here](https://dl.acm.org/doi/abs/10.1145/3539618.3592048).

## Getting Started

### Creating the Environment
If you use conda you can create the env using the **environment.yml** file
(depending on the hardware some versions might need to be different):

`conda env create -f environment.yml`

Activate environment with:

`conda activate rating_prediction_env`

### Data
In the paper, to evaluate our models, we use data collected in the first edition of the [Alexa Prize TaskBot challenge](https://www.amazon.science/alexa-prize/proceedings/twiz-a-conversational-task-wizard-with-multimodal-curiosity-exploration).\
The data cannot be shared due to Alexa Prize Challenge rules, but you can provide your own data and train the models.

#### Data Format
You can provide data in the format availble in [input_example_rating_prediction.json](input_example_rating_prediction.json) (a dummy example).\
The format is the following:
```
{
  "an_id_of_a_session": {   # this one session
    "session_id": "an_id_of_a_session",
    "user_id": "an_id_of_a_user",
    "conversation_id": "an_id_of_a_conversation",
    "start_time": "2023-01-17T04:00:18.592746",  # session start time
    "rating_timestamp": "",  # rating time
    "duration": "250",  # time in seconds of the session
    "rating": "5",   # a str number from 1 to 5 or none if no rating was given
    "has_screen": false,  # bool indicating if the device had a screen
    "turns": [  # list of turns
      {
        "creation_date_time": "2023-01-17T04:00:59.347709",  # turn start time
        "candidate_responses": {  # dict with the candidate RG names and their response
          "LAUNCH_RESPONDER": {
            "response": " Try asking me things like: How to cook grilled salmon, or, How to make a tote bag. "
          }
        },
        "intent": "YesIntent",  # intent as classified by an external method
        "sentiment_score": {  # sentiment score as given by Alexa Skills Kit
          "valence": {
            "value": "0.020472467",
            "confidence": "UNKNOWN"
          },
          "activation": {
            "value": "0.083258376",
            "confidence": "UNKNOWN"
          },
          "satisfaction": {
            "value": "0.012997696",
            "confidence": "UNKNOWN"
          },
          "modelVersion": "2.1"
        },
        "session_id": "an_id_of_a_session",
        "asr": [  # ASR scores as given by Alexa Skills Kit
          {
            "tokens": [
              {
                "startOffsetInMilliseconds": "460",
                "endOffsetInMilliseconds": "610",
                "_confidenceV2": {
                  "score": "0.988"
                },
                "value": "yes",
                "confidence": "0.988"
              }
            ],
            "_confidenceV2": {
              "score": "0.988"
            },
            "directedness": {
              "score": "0"
            },
            "confidence": "0.988"
          }
        ],
        "data_source": "",   # source of the task (e.g. recipe, wikihow, or empty)
        "system_latency": 1000,  # time it takes for the system to respond
        "conversation_id": "an_id_of_a_conversation",
        "text": "yes",   # user utterance
        "phase": "greeting",  # current phase of the dialogue (e.g. greeting, process_grounding, steps)
        "user_id": "an_id_of_a_user",
        "input_offensive": "0",   # 0 or 1 indicating offensive content as given by an Alexa internal method and by matching with a list of sensitive words
        "response": " Try asking me things like: How to cook grilled salmon, or, How to make a tote bag. "  # system response
       }
        ... # it can have more turns
     ],
  }
  ... # it can have more sessions
}
```


If you do not have access to some of these fields, or do not want to consider them check the next section.

#### Creating and Adapting new Features
To change the input format adapt the classes [AllInteractions](./data_binding/interactions.py), [Session](./data_binding/session.py), and [Turn](./data_binding/turn.py).

To change the behavior features considered choose the ones you want from the [./models/session_features.py](./models/session_features.py) 
in the function *get_features_from_turn*.


## Rating Prediction Models
Models available: 
* **Behavior-Only** - methods based only on behavior features such as RandomForest, AdaBoost, Bagging, GradientBoosting, XGBoost, LogisticRegression, and SVM.
* **Conversarional-Flow-Only** - methods that only consider text features such as BERT and T5.
* **Conversational-Flow and Behavior** - the proposed TB-Rater model.


### Training and Evaluation
To train and evaluate the model use the [train_eval.py](train_eval.py) script.

For example to run the default BERT model:\
`python3 train_eval.py --model_type bert`

You can also give a path to a json file with a model configuration: \
`python3 train_eval.py --model_type bert --config ./run_configurations/basic_bert.json`

In the [run_configurations](./run_configurations) folder there are some example configurations.


In the end this creates in the provided **output_folder** a folder for the model checkpoints,
where each one has: config.json, pytorch .bin model, traininer_state.json and optimizer parameters, as explained in Huggingface's Trainer documentation.

Adding to the checkpoints the script also creates:
* **tokenizer** - a folder with the tokenizer information.
* **_config.json** - a file with the model configuration used to train the model which can then be used to load the model for other purposes.
* **_metrics.json** - a file with results obtained.
* **_all.json** - a file which combines config and metrics in a single file for easier processing.
* **_predictions.tsv** - a file with the model predictions and the main dialogue attributes for analysis.

### Creating your own Models

#### Using a Custom Configuration
You can create your own model by providing a different configuration json file. \
The parameters available for each model are a combination of the variables at: 
* [./models/model_utils.py](./models/model_utils.py) in the DefaultModelConfiguration class
* With a corresponding model architecture configuration class [./models](./models) (a class that inherits from DefaultModelConfiguration).


#### Creating a New Model from Scratch
To create a new architecture from scratch, i.e., one not present in the paper, you can follow a similar approach 
to the ones in [./models](./models):
1. Add the new model name configuration to the Enum class *RatingPredictionModels* in [models_utils.py](./models/model_utils.py).
2. Create a new python file and create a class that inherits from *DefaultModelConfiguration* and add your new default values and/or new attributes.
3. Create the appropriate functions to *load_model_tokenizer*, *get_predictions* and *train_eval_model*.
4. Add the new model to [train_eval.py](train_eval.py) using another *elif* that calls the *train_eval_model* function.


#### Running the Custom Configuration/Model
After having a custom configuration file or model run:\
`python3 train_eval.py --model_type <model_type> --config <path_to_config>`


## Citation
If you find it useful please cite our work using:
```
@inproceedings{rating_prediction_cta,
author = {Ferreira, Rafael and Semedo, David and Magalh\~{a}es, Jo\~{a}o},
title = {Rating Prediction in Conversational Task Assistants with Behavioral and Conversational-Flow Features},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3592048},
doi = {10.1145/3539618.3592048},
abstract = {Predicting the success of Conversational Task Assistants (CTA) can be critical to understand user behavior and act accordingly. In this paper, we propose TB-Rater, a Transformer model which combines conversational-flow features with user behavior features for predicting user ratings in a CTA scenario. In particular, we use real human-agent conversations and ratings collected in the Alexa TaskBot challenge, a novel multimodal and multi-turn conversational context. Our results show the advantages of modeling both the conversational-flow and behavioral aspects of the conversation in a single model for offline rating prediction. Additionally, an analysis of the CTA-specific behavioral features brings insights into this setting and can be used to bootstrap future systems.},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2314–2318},
numpages = {5},
keywords = {rating prediction, nlp, conversational task assistants},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```
