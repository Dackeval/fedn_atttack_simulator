from load__data import load_data
from model import load_parameters, save_metrics
from sklearn.metrics import log_loss, accuracy_score
import os
import sys
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)



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

    eps = 1e-15  # numerical instability

    train_proba = model.predict_proba(x_train)
    train_proba = np.clip(train_proba, eps, 1 - eps)  # ensures values are in valid range
    train_proba /= train_proba.sum(axis=1, keepdims=True)

    test_proba = model.predict_proba(x_test)
    test_proba = np.clip(test_proba, eps, 1 - eps)
    test_proba /= test_proba.sum(axis=1, keepdims=True)

    report = {
        "training_loss": log_loss(y_train, train_proba, labels=[0, 1, 2]),
        "training_accuracy": accuracy_score(y_train, model.predict(x_train)),
        "test_loss": log_loss(y_test, test_proba, labels=[0, 1, 2]),
        "test_accuracy": accuracy_score(y_test, model.predict(x_test))
    }

    logger.info('Training accuracy: %.4f', accuracy_score(y_train, model.predict(x_train)))
    logger.info('Training loss: %.4f', log_loss(y_train, train_proba, labels=[0, 1, 2]))
    logger.info('Test accuracy: %.4f', accuracy_score(y_test, model.predict(x_test)))
    logger.info('Test loss: %.4f', log_loss(y_test, test_proba, labels=[0, 1, 2]))   

    save_metrics(report, out_json_path)
    logger.info('Validation Completed!')
    return report

if __name__ == "__main__":
    # usage: python validate.py in_model_path out_json_path [data_path]
    in_model = sys.argv[1]
    out_json = sys.argv[2]
    data_path = None
    if len(sys.argv) > 3:
        data_path = sys.argv[3]
    validate(in_model, out_json, data_path)
