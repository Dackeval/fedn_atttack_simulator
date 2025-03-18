from load__data import load_data
from model import load_parameters, save_metrics
from sklearn.metrics import log_loss, accuracy_score
import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from load_environment_param import load_env_params

logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)

def validate(global_model, out_json_path='/app/validation.json', data_path=None, malicious=False, attack=None):
    """ Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    x_train, y_train = load_data()
    x_test, y_test = load_data(is_train=False)
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    # load the local model for train acc and loss
    client_index = os.environ.get("CLIENT_ID", "1")
    local_model = load_parameters(f"/app/model_update_{client_index}.npz")
    logger.info(f"/app/model_update_{client_index}.npz")


    eps = 1e-15  # numerical instability
    train_proba = local_model.predict_proba(x_train)
    train_proba = np.clip(train_proba, eps, 1 - eps)

    row_sums = train_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = eps
    train_proba /= row_sums


    test_proba = global_model.predict_proba(x_test)
    test_proba = np.clip(test_proba, eps, 1 - eps)
    row_sums = test_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = eps
    test_proba /= row_sums

    report = {
        "training_loss": log_loss(y_train, train_proba, labels=[0, 1, 2]),
        "training_accuracy": accuracy_score(y_train, local_model.predict(x_train)),
        "test_loss": log_loss(y_test, test_proba, labels=[0, 1, 2]),
        "test_accuracy": accuracy_score(y_test, global_model.predict(x_test))
    }

    logger.info('Training accuracy: %.4f', accuracy_score(y_train, local_model.predict(x_train)))
    logger.info('Training loss: %.4f', log_loss(y_train, train_proba, labels=[0, 1, 2]))
    logger.info('Test accuracy: %.4f', accuracy_score(y_test, global_model.predict(x_test)))
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
