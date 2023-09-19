from collections import Counter
import seaborn as sns
import json
import os
from typing import Dict, List, Union, Tuple, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, \
    max_error, mean_squared_error, mean_absolute_error, r2_score
from data_binding.interactions import AllInteractions
from dataset.dataset import LabellingMethods, split_dataset, convert_conversation_label, \
    labeling_method_to_number_of_labels, DATASET_SEED
from models.model_utils import class_attributes_to_dict, set_seeds, BehaviourOnlyModelConfiguration
from models.session_features import ScalingMethod, SessionFeaturesVectors
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, \
    BaggingClassifier, GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from enum import Enum
import xgboost as xgb
from matplotlib import pyplot as plt
import matplotlib


# avoid type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()


class BehaviourOnlyModels(Enum):
    RandomForestClassifier = "RandomForestClassifier"
    AdaBoostClassifier = "AdaBoostClassifier"
    BaggingClassifier = "BaggingClassifier"
    GradientBoostingClassifier = "GradientBoostingClassifier"
    LogisticRegression = "LogisticRegression"
    XGBoost = "XGBoost"
    SVM = "SVM"


def train_eval_model(configuration: Dict):
    # train model and evaluate on validation and test sets

    # get the model configuration
    model_configuration = BehaviourOnlyModelConfiguration(configuration)
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

    number_turns_consider = 2 if model_configuration.use_penultimate else 1
    train_session_vectors = SessionFeaturesVectors(sessions_list=s_train, number_turns=number_turns_consider,
                                                   scaling=scalling_method)
    train_feature_vectors = train_session_vectors.sessions_behaviour_features

    val_session_vectors = SessionFeaturesVectors(sessions_list=s_val, number_turns=number_turns_consider,
                                                 scaling=scalling_method)
    val_feature_vectors = val_session_vectors.sessions_behaviour_features

    test_session_vectors = SessionFeaturesVectors(sessions_list=s_test, number_turns=number_turns_consider,
                                                  scaling=scalling_method)
    test_feature_vectors = test_session_vectors.sessions_behaviour_features

    # feature vectors are List[List[List[float]]]  ---->   session / turn / features
    # remove the turn dimension because there is only one turn
    train_feature_vectors = np.squeeze(np.asarray(train_feature_vectors)[:, 0, :])
    val_feature_vectors = np.squeeze(np.asarray(val_feature_vectors)[:, 0, :])
    test_feature_vectors = np.squeeze(np.asarray(test_feature_vectors)[:, 0, :])

    # go through all classifier types
    results = []
    predictions_dict = {}
    if model_configuration.test_models in ["all", None]:
        # test with all the models
        iterable_models = BehaviourOnlyModels._member_names_
    else:
        # test only with the models passed
        iterable_models = model_configuration.test_models

    for i, classifier_type in enumerate(iterable_models):
        # set seeds for reproducibility
        set_seeds(model_configuration.seed)

        print(f"({i + 1}/{len(iterable_models)}) Training:", classifier_type)
        behaviour_only_model = BehaviourOnlyModels(classifier_type)
        trained_clf, best_params, val_score, test_score = train_sklearn_ensemble(
            behaviour_only_model=behaviour_only_model,
            train_feature_vectors=train_feature_vectors,
            val_feature_vectors=val_feature_vectors,
            test_feature_vectors=test_feature_vectors,
            l_train=l_train, l_val=l_val, l_test=l_test,
            labelling_method=labelling_method,
            seed=model_configuration.seed,
            is_regression=labelling_method == LabellingMethods.Regression,
            use_predefined_validation=model_configuration.use_predefined_validation_split,
            metric_for_best_model=model_configuration.metric_for_best_model
        )

        test_predictions = trained_clf.predict(test_feature_vectors)

        results.append({
            "classifier_type": classifier_type,
            "best_params": best_params,
            "val_score": val_score,
            "test_score": test_score,
            "scaling": scalling_method.value if scalling_method else scalling_method,
            "labelling_method": labelling_method.value,
            "using_penultimate": model_configuration.use_penultimate,
            "seed": model_configuration.seed,
            "use_predefined_validation_split": model_configuration.use_predefined_validation_split,
            "number_features": train_feature_vectors.shape[-1],  # infer size by checking this
            "data_location": model_configuration.data_location,
            "metric_for_best_model": model_configuration.metric_for_best_model
        })

        if labelling_method == LabellingMethods.Regression:
            other_metrics = {
                "max_error": max_error(y_pred=test_predictions, y_true=l_test),
                "mean_squared_error": mean_squared_error(y_pred=test_predictions, y_true=l_test),
                "mean_absolute_error": mean_absolute_error(y_pred=test_predictions, y_true=l_test),
                "r2_score": r2_score(y_pred=test_predictions, y_true=l_test)
            }
        else:
            other_metrics = {
                "test_precision": precision_score(y_pred=test_predictions, y_true=l_test, average="macro"),
                "test_recall": recall_score(y_pred=test_predictions, y_true=l_test, average="macro"),
                "test_f1": f1_score(y_pred=test_predictions, y_true=l_test, average="macro"),
                "test_accuracy": accuracy_score(y_pred=test_predictions, y_true=l_test),
                "predictions_count": {int(k): int(v) for k, v in Counter(test_predictions).items()},
                "labels_count": {int(k): int(v) for k, v in Counter(l_test).items()},
            }

        # add the other metrics
        results[-1].update(other_metrics)

        # just print
        for key, value in results[-1].items():
            print(key, value)

        # checks if the param exists before attempting to plot importance
        os.makedirs(model_configuration.visualization_dir, exist_ok=True)
        plot_feature_importance(trained_clf=trained_clf,
                                feature_names=train_session_vectors.session_features_names[0],  # 0 because of batch
                                clf_name=classifier_type,
                                save_fig_path=os.path.join(model_configuration.visualization_dir,
                                                           f"{classifier_type}_feature_importance.pdf")
                                )

        for current_session, prediction in zip(s_test, test_predictions):
            predictions_dict[current_session.session_id] = {
                "original_rating": current_session.rating,
                "label_rating": convert_conversation_label(method=labelling_method, label_value=current_session.rating),
                "predicted_rating": int(prediction),
                "session": current_session.session_to_simple_dict(),
            }

        output_path = os.path.join(model_configuration.output_dir, f"{classifier_type}_predictions.json")
        os.makedirs(model_configuration.output_dir, exist_ok=True)
        with open(output_path, "w") as f_open:
            print(f"Predictions for {classifier_type} written to {output_path}:")
            print()
            json.dump(predictions_dict, f_open, indent=2, default=np_encoder)

    output_path = os.path.join(model_configuration.output_dir, f"{model_configuration.suffix}_results_summary.json")
    os.makedirs(model_configuration.output_dir, exist_ok=True)
    with open(output_path, "w") as f_open:
        print("Results Summary:")
        print(results)
        print()
        json.dump(results, f_open, indent=2, default=np_encoder)


def train_sklearn_ensemble(behaviour_only_model: BehaviourOnlyModels,
                           train_feature_vectors: Union[List[float], np.ndarray],
                           val_feature_vectors: Union[List[float], np.ndarray],
                           test_feature_vectors: Union[List[float], np.ndarray],
                           l_train: List[int], l_val: List[int],
                           l_test: List[int], labelling_method: LabellingMethods,
                           seed: int, is_regression: bool,
                           use_predefined_validation: bool,
                           metric_for_best_model: Union[str, None]) -> Tuple[Any, Any, float, float]:
    # optimize params
    if behaviour_only_model == BehaviourOnlyModels.GradientBoostingClassifier:
        parameters = {"n_estimators": np.arange(50, 500, 50), "learning_rate": np.arange(0.1, 2.0, 0.3),
                      "max_depth": np.arange(1, 10, 2), "random_state": [seed]}
        if is_regression:
            clf_model = GradientBoostingRegressor(random_state=seed)
        else:
            clf_model = GradientBoostingClassifier(random_state=seed)
    elif behaviour_only_model == BehaviourOnlyModels.RandomForestClassifier:
        parameters = {"n_estimators": np.arange(50, 500, 50), "max_depth": np.arange(1, 10, 1), "random_state": [seed]}
        if is_regression:
            clf_model = RandomForestRegressor(random_state=seed)
        else:
            clf_model = RandomForestClassifier(random_state=seed)
    elif behaviour_only_model == BehaviourOnlyModels.AdaBoostClassifier:
        parameters = {"n_estimators": np.arange(50, 500, 50), "learning_rate": np.arange(0.1, 2.0, 0.2),
                      "random_state": [seed]}
        if is_regression:
            clf_model = AdaBoostRegressor(random_state=seed)
        else:
            clf_model = AdaBoostClassifier(random_state=seed)
    elif behaviour_only_model == BehaviourOnlyModels.BaggingClassifier:
        parameters = {"n_estimators": np.arange(50, 500, 50), "random_state": [seed]}
        if is_regression:
            clf_model = BaggingRegressor(base_estimator=SVC(), random_state=seed)
        else:
            clf_model = BaggingClassifier(base_estimator=SVC(), random_state=seed)
    elif behaviour_only_model == BehaviourOnlyModels.XGBoost:
        parameters = {'n_estimators': np.arange(50, 500, 100), 'colsample_bytree': [0.7, 0.8], 'max_depth': [1, 10, 2],
                      'reg_alpha': [1.1, 1.3], 'reg_lambda': [1.1, 1.3], 'subsample': [0.7, 0.9]}
        if is_regression:
            clf_model = xgb.XGBRegressor(seed=seed)
        else:
            eval_metric = "error" if labeling_method_to_number_of_labels(labelling_method) == 2 else "merror"
            clf_model = xgb.XGBClassifier(eval_metric=eval_metric, seed=seed)
    elif behaviour_only_model == BehaviourOnlyModels.LogisticRegression:
        if is_regression:
            parameters = {}
            clf_model = LinearRegression()
        else:
            parameters = {'C': [.001, .01, .1, 1, 10, 100, 1000]}
            clf_model = LogisticRegression(random_state=seed, max_iter=10000)
    elif behaviour_only_model == BehaviourOnlyModels.SVM:
        parameters = {'C': [.001, .01, .1, 1, 10, 100, 1000]}
        if is_regression:
            clf_model = svm.SVR()
        else:
            clf_model = svm.SVC(random_state=seed)
    else:
        raise ValueError(f"behaviour_only_model param {behaviour_only_model} not recognized.")

    new_train_feature_vectors = np.concatenate((train_feature_vectors, val_feature_vectors), axis=0)
    new_l_train = np.concatenate((l_train, l_val), axis=0)

    # create gridsearch
    if not use_predefined_validation:
        # since GridSearchCV automatically uses 5-fold cross-validation we can join the train and val sets together
        clf = GridSearchCV(clf_model, parameters, refit=True)
    else:
        split_index = [-1] * len(train_feature_vectors) + [0] * len(val_feature_vectors)
        pds = PredefinedSplit(test_fold=split_index)
        clf = GridSearchCV(estimator=clf_model, param_grid=parameters, refit=True, cv=pds,
                           scoring=metric_for_best_model if behaviour_only_model != BehaviourOnlyModels.XGBoost else None)

    # fit to data
    clf = clf.fit(new_train_feature_vectors, new_l_train)

    # get score
    val_score = clf.score(val_feature_vectors, l_val)
    test_score = clf.score(test_feature_vectors, l_test)

    return clf.best_estimator_, clf.best_params_, float(val_score), float(test_score)


def plot_feature_importance(trained_clf, feature_names: List[str], clf_name: str = None, save_fig_path: str = None,
                            top_n_features: int = 14, use_title: bool = False, log_scale: bool = False):
    # trained_clf is a sklearn model
    if getattr(trained_clf, "feature_importances_", None) is not None:
        sorted_idx = trained_clf.feature_importances_.argsort()
        fig = plt.figure()
        fig.tight_layout()

        # color map
        c = sns.color_palette("ch:start=.2,rot=-.3", top_n_features if top_n_features else len(feature_names))

        plt.barh(np.array(feature_names)[sorted_idx][-top_n_features:], trained_clf.feature_importances_[sorted_idx][-top_n_features:], color=c, log=log_scale)
    elif getattr(trained_clf, "coef_", None) is not None:
        # onyly for linear kernel in SVM
        sorted_idx = trained_clf.coef_.argsort()[0]
        fig = plt.figure()
        fig.tight_layout()
        if top_n_features % 2 != 0:
            top_n_features += 1

        # color map
        c = sns.color_palette("ch:start=.2,rot=-.3", top_n_features if top_n_features else len(feature_names))

        # get the half from the top and half from the last
        top_n_features = int(top_n_features / 2)
        top_x = [np.array(feature_names)[sorted_idx][-top_n_features:], trained_clf.coef_[0][sorted_idx][-top_n_features:]]
        bottom_x = [np.array(feature_names)[sorted_idx][:top_n_features], trained_clf.coef_[0][sorted_idx][:top_n_features]]
        plt.barh(np.concatenate((bottom_x[0], top_x[0])), np.concatenate((bottom_x[1], top_x[1])), color=c, log=log_scale)
    else:
        return 0

    # plt.xlabel("Feature Importance", fontsize=14)
    # plt.ylabel("Feature Name", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)

    if use_title and clf_name:
        plt.title(clf_name, fontsize=18)

    plt.tight_layout()
    plt.plot()
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()
