import pandas as pd
import numpy as np
import sys
from model import load_parameters, save_parameters
import os
import json
from data import load_data
from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

HELPER_MODULE = 'numpyhelper'
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 10

def train(in_model_path, out_model_path, data_path=None, malicious=False, attack=None, batch_size=130, epochs=1):
    """ Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """
    # Load data
    x_train, y_train = load_data(data_path)

    # Implement different version of data loading for malicious clients
    if malicious:
        match attack:
            case 'grad_boost_basic':
                ### Gradient inflation attack ###
                # DO NOTHING IN THE DATA LOADING PROCESS
                print("DO NOTHING IN THE DATA LOADING PROCESS")
                ### End of inflation attack code ###
            case 'label_flip_basic':
                ### Label flipping attack - basic
                y_train_unflipped = y_train
                L = 3 # of labels
                y_train = [L - 1 - y for y in y_train]
                print('Running a label flip :D')
                ### End of label flipping attack - basic
            case None:
                print('No attack was specified for the malicious client.')
            case _:
                print("DO NOTHING!")

    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)

    # Implement different version of training for malicious clients
    if malicious:
        match attack:
            case 'grad_boost_basic':
                ### Gradient inflation attack ###
                print(f"Coefs before boost: {model.coef_}")
                inflation_factor = 100  # Can be adjusted
                print(f"A boost factor of {inflation_factor} is applied on the parameters!")
                model.coef_ = inflation_factor * model.coef_
                model.intercept_ = inflation_factor * model.intercept_
                print(f"Coefs after boost: {model.coef_}")
                ### End of inflation attack code ###
            case 'label_flip_basic':
                ### Label flipping attack - basic
                print("DO NOTHING IN THE TRAINING PROCESS")
                ### End of label flipping attack - basic
            case 'little_is_enough':
                ### Little-Is-Enough attack - basic
                print("Don't do anything!")
                ### End of Little-Is-Enough attack - basic
            case None:
                print('No attack was specified for the malicious client.')

    model.fit(x_train, y_train.values.ravel())

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        'num_examples': len(x_train),
        'epochs': model.get_params()['max_iter']
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)

    # #params_path = f"/var/parameters/{os.uname().nodename}"
    params_path = "parameters"
    if not os.path.exists(params_path):
        os.makedirs(params_path)

    params_json_path = os.path.join(params_path, "params.json")

    # Check if params.json already exists:
    if os.path.exists(params_json_path):
        # File exists, so read and update
        with open(params_json_path, "r") as json_file:
            params_json = json.load(json_file)
        params_json['local_params'].append(
            np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1).tolist()
        )
    else:
        # File does not exist yet, so create and initialize
        params_json = {
            "local_params": [
                np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1).tolist()
            ],
            "global_params": []
        }

    # Now write the file (in either case)
    with open(params_json_path, "w") as json_file:
        json.dump(params_json, json_file)

if __name__ == "__main__":
    """
    Example usage:
    python train.py <in_model_path> <out_model_path> [<data_path> [<batch_size> <epochs> <lr> <malicious> <attack>]]
    """
    in_model = sys.argv[1]
    out_model = sys.argv[2]

    data_path = None
    if len(sys.argv) > 3:
        data_path = sys.argv[3]

    # parse optional arguments as needed
    batch_size = 130
    epochs = 1
    malicious_flag = False
    attack_type = None

    if len(sys.argv) > 4:
        batch_size = int(sys.argv[4])
    if len(sys.argv) > 5:
        epochs = int(sys.argv[5])
    if len(sys.argv) > 6:
        # "True" or "False"
        malicious_flag = (sys.argv[6].lower() == "true")
    if len(sys.argv) > 7:
        attack_type = sys.argv[7]

    train(in_model, out_model, data_path, batch_size, epochs, malicious_flag, attack_type)
