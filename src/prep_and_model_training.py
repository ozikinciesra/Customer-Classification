import argparse
import os
import pickle
from collections import Counter

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler


class ClassifierModel:
    def __init__(self, dependent_variable="has_pod"):
        self.dependent_variable = dependent_variable

    def write_pickle(self, obj, *path):
        """
        Filepath in .pkl
        """
        path = self.construct_path(*path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def construct_path(self, *path):
        path = os.path.join(*path)
        path = (os.path.join(os.path.dirname(__file__), path)
                if not os.path.isabs(path)
                else path)
        return path

    def read_pickle(self, *path):
        path = self.construct_path(*path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def feature_scaling(self, type, df, columns):
        """Generate ML Pipeline which include both pre-processing and model training.
        """
        path_pickle = "../model/scaled_data.pkl"

        if type == 'train':
            mm = MinMaxScaler()
            mm.fit(df[columns])
            print("Info: Saving Feature Scaling Pipeline to following path: " + str(path_pickle))
            self.write_pickle(mm, path_pickle)
        if type == 'predict':
            print("Info: Reading Feature Scaling Pipeline from following path: " + str(path_pickle))
            mm = self.read_pickle(path_pickle)
        X_train_scaled = mm.transform(df[columns])
        X_train_scaled = pd.DataFrame(X_train_scaled)
        X_train_scaled.columns = columns
        return X_train_scaled

    def train_model(self, train_set, train_labels):
        print("Info: Model training started with: " + str(train_set.shape[0]) + " sample")
        model = xgb.XGBClassifier(learning_rate=0.01, min_child_weight=8, eta=0.2)
        print("Info: Model Parameters: " + str(model.get_params()))
        model.fit(train_set, train_labels)
        pickle_path = "../model/trained_data.pkl"
        self.write_pickle(model, pickle_path)
        print("Info: Model trained and pickled at " + str(pickle_path))
        return model

    def split_train_test_validate(self, model_data):
        print("Info: Split dataset with StratifiedShuffleSplit methods: ")
        # to able to do cross validation I am using StratifiedShuffleSplit methods instead of train_test_split from sklearn.model_selection
        # for that sampledataset, cross validation hasn't done, so no need validate dataset
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=9)
        for train_index, validate_test_index in split.split(model_data, model_data[self.dependent_variable]):
            train_set = model_data.iloc[train_index]
            test_set = model_data.iloc[validate_test_index]
        return train_set, test_set

    def eval_metrics(self, actual, pred):
        cnf_matrix = confusion_matrix(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        metrics = {"accuracy": accuracy,
                   "precision": precision,
                   "recall": recall,
                   "f1": f1}
        return metrics, cnf_matrix

    def run_model(self, df_features, features):
        print("Info: Running model started")
        X_columns = features
        train_set, test_set = self.split_train_test_validate(df_features)
        train = self.feature_scaling("train", train_set.reset_index(drop=True), columns=features)
        train_label = train_set[self.dependent_variable].reset_index(drop=True).copy()
        test = self.feature_scaling("predict", test_set.reset_index(drop=True), columns=features)
        test_label = test_set[self.dependent_variable].reset_index(drop=True).copy()

        print("Info: Train mode counter before SMOTE : ", Counter(train_label))
        # Do over and under sampling because we have unbalanced data
        over = SMOTE(sampling_strategy=0.2)
        under = RandomUnderSampler(sampling_strategy=0.12)
        steps = [('u', under), ('o', over)]

        pipeline = Pipeline(steps=steps)
        # transform the dataset
        train, train_label = pipeline.fit_resample(train, train_label)
        print("Info: Train mode counter after SMOTE : ", Counter(train_label))

        train_set = train[X_columns]
        test_set = test[X_columns]
        model = self.train_model(train_set, train_label)

        # Add prediction column on the train dataset
        y_hat_train = model.predict(train)
        train[self.dependent_variable] = train_label
        train["pred_label"] = y_hat_train

        # Add prediction column on the test dataset
        y_hat_test = model.predict(test)
        test[self.dependent_variable] = test_label
        test["pred_label"] = y_hat_test

        print("eval metrics for train dataset: ", self.eval_metrics(train_label, y_hat_train)[0])
        print("eval metrics for test dataset: ", self.eval_metrics(test_label, y_hat_test)[0])
        return train, test

    def create_new_features(self, df):
        # create new count columns:
        df["total_count_computer_phone_tablet"] = df["computer_count"] + df["tablet_count"] + df["mobile_phones_count"]
        df["total_count"] = df["computer_count"] + df["tablet_count"] + df["mobile_phones_count"] + df["other_count"] + \
                            df["smarthome_count"] + df["wearables_count"] + df["gaming_count"] + df[
                                "tv_&_audio_count"] + df["printer_&_scanners_count"]

        df.loc[df["traffic_tx_bucket"] <= 2, "bad_quality_traffic_tx"] = 1
        df.loc[df["traffic_rx_bucket"] <= 3, "bad_quality_traffic_rx"] = 1
        df.loc[df["speed_tier_bucket"] <= 3, "bad_quality_speed_tier"] = 1
        df.loc[df["interference_5G_bucket"] < 2, "bad_quality_interference_5G"] = 1
        df.loc[df["WEM_bucket"] <= 2, "bad_quality_WEM"] = 1

        df = df.fillna(0)
        return df


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Process for building model.")
        parser.add_argument("--input_file",
                            dest="input_file",
                            required=False,
                            default="../raw_data/sample_input_data.csv",
                            help="Input file path")
        parser.add_argument("--target_column",
                            dest="target_column",
                            required=False,
                            default="has_pod",
                            help="load type either model_training or prediction")
        args = parser.parse_args()

        dependent_column = args.target_column
        file_path = args.input_file
        classifier_model = ClassifierModel(dependent_variable=dependent_column)
        df = pd.read_csv(file_path, sep=",")
        df = classifier_model.create_new_features(df)

        # selected by using Initial Analyses.ipynb jupyter notebook script
        selected_features = ["total_count_computer_phone_tablet", "total_count", "tablet_count", "smarthome_count",
                             "total_count_gaming_tv_printer", "speed_tier_bucket", "bad_quality_speed_tier",
                             "bad_quality_interference_5G", "bad_quality_WEM"]

        train_df, test_df = classifier_model.run_model(df, features=selected_features)

        print("Info: Completed model training")
    except Exception as e:
        print("Error in building the model ", "", classifier_model.__class__.__name__)
        print(e)
        raise
