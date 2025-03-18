import pandas as pd
import numpy as np
import sys
from model import save_parameters
import os
import json
from load__data import load_data
from fedn.utils.helpers.helpers import get_helper, save_metadata
import logging
from load_environment_param import load_env_params


HELPER_MODULE = 'numpyhelper'
helper = get_helper(HELPER_MODULE)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fedn")

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def train(model):
    """ Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    """
    # Load environment variables, print them and set output model path
    client_index_str, malicious, attack, inflation_factor, batch_size, _, _, _, _ = load_env_params()
    out_model_path = f"/app/model_update_{client_index_str}.npz"    
    logger.info(f"out_model_path={out_model_path}")
    logger.info(f"[TRAIN] client_index={client_index_str}, malicious={malicious}, attack={attack}")
    logger.info(f"[TRAIN] hyperparams: epochs={20}, batch_size={batch_size}, inflation_factor={inflation_factor}")

    # load data
    x_train, y_train = load_data()
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if malicious:
        logger.info(f"Attack mode '{attack}' enabled.")
        match attack:
            case 'grad_boost_basic':
                model.coef_ = inflation_factor * model.coef_
                model.intercept_ = inflation_factor * model.intercept_            
            case 'label_flip_basic':
                L = 3 # number of classes in IRIS
                y_train = (L - 1 - y_train).tolist()            
            case None:
                logger.info("[Attack] Attack was set to None, ignoring.")
            case _:
                logger.warning(f"[Attack] Unrecognized attack: {attack}")


    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train)

    n_samples = len(x_train_df)
    rng = np.random.default_rng(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    x_train_df = x_train_df.iloc[indices].reset_index(drop=True)
    y_train_df = y_train_df.iloc[indices].reset_index(drop=True)

    model.fit(x_train_df, y_train_df.values.ravel())

    metadata = {
        # num_examples are mandatory
        'num_examples': len(x_train),
        'epochs': 20 # epochs are predefined static in the model.py as 20, set through max_iter
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)

    params_path = "paramaters"
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

    with open(params_json_path, "w") as json_file:
        json.dump(params_json, json_file)
    
    logger.info('Train Completed!')
    return metadata, model


