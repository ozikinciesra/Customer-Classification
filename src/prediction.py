import argparse

import numpy as np
import pandas as pd

from src.prep_and_model_training import ClassifierModel

parser = argparse.ArgumentParser(description="Process for building model.")
parser.add_argument("-i",
                    "--prediction_file",
                    dest="prediction_file",
                    required=False,
                    default="../raw_data/sample_prediction_data.csv",
                    help="Input file path")
args = parser.parse_args()
prediction_file_path = args.prediction_file
df_predict = pd.read_csv(prediction_file_path, sep=",")

features = ["total_count", "bad_quality_speed_tier", "total_count_gaming_tv_printer", "tablet_count",
            "smarthome_count", "total_count_computer_phone_tablet", "bad_quality_WEM", "wearables_count",
            "gaming_count", "bad_quality_interference_5G"]
df_predict = ClassifierModel().create_new_features(df_predict)
model_path = "../model/trained_data.pkl"
xgbr_model = ClassifierModel().read_pickle(model_path)

if "has_pod" in df_predict.columns:
    df_predict = df_predict.drop(columns="has_pod")
X = ClassifierModel().feature_scaling("predict", df_predict, columns=features)
df_predict.loc[:, "pred_pod"] = xgbr_model.predict(X[features])
df_predict.loc[:, "pred_pod"] = np.rint(df_predict["pred_pod"])
pred_prob_xgbr = xgbr_model.predict_proba(X[features])
print(df_predict[df_predict["pred_pod"] > 0]["gw_mac_address"].values)
