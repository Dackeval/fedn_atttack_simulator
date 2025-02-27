from data import load_data
from model import load_parameters, save_metrics
from sklearn.metrics import log_loss, accuracy_score
import os
import sys
import pandas as pd
import numpy as np
import json



def validate(model, out_json_path='/app/validation.json', data_path=None, malicious=False, attack=None):
    """ Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    x_train, y_train = load_data(data_path)
    x_test, y_test = load_data(data_path, is_train=False)
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    # Load model
    #model = load_parameters(in_model_path)

    # params_path = f"/var/parameters/{os.uname().nodename}"
    params_path = "parameters"
    if not os.path.exists(params_path):
        os.makedirs(params_path)

    params_json_path = os.path.join(params_path, "params.json")

    with open(f"{params_path}/params.json", "r") as json_file:
        params_json = json.load(json_file)
        params_json['global_params'].append(np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1).tolist())

    with open(f"{params_path}/params.json", "w") as json_file:
        json.dump(params_json, json_file)

   # JSON schema
    report = {
        "training_loss": log_loss(y_train, np.nan_to_num(model.predict_proba(x_train), 1), labels=[0, 1, 2]),
        "training_accuracy": accuracy_score(y_train, np.nan_to_num(model.predict(x_train),0)),
        "test_loss": log_loss(y_test, np.nan_to_num(model.predict_proba(x_test), 1), labels=[0, 1, 2]),
        "test_accuracy": accuracy_score(y_test, np.nan_to_num(model.predict(x_test), 0))
    }

    # Save JSON
    save_metrics(report, out_json_path)
    print("Validation complete.")
    return report

if __name__ == "__main__":
    # usage: python validate.py in_model_path out_json_path [data_path]
    in_model = sys.argv[1]
    out_json = sys.argv[2]
    data_path = None
    if len(sys.argv) > 3:
        data_path = sys.argv[3]
    validate(in_model, out_json, data_path)
