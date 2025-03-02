import pandas as pd
import numpy as np
import sys
from model import load_parameters, save_parameters
import os
import json
from load__data import load_data
from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics
import logging



HELPER_MODULE = 'numpyhelper'
helper = get_helper(HELPER_MODULE)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fedn")

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def train(model, out_model_path='/app/model_update.npz',  
          data_path=None,batch_size=32, epochs=1, malicious=False, attack=None):
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
    client_index = os.environ.get("CLIENT_INDEX", "1")
    out_model_path = f"/app/model_update_{client_index}.npz"
    logger.info(f"out_model_path={out_model_path}")
    # Load data
    # Check if container sets MALICIOUS=true
    env_malicious_flag = os.environ.get("MALICIOUS", "false").strip().lower()
    # Convert to boolean
    env_malicious = (env_malicious_flag == "true")

    logger.info(f"env_malicious_flag={env_malicious_flag}, env_malicious={env_malicious}")
    client_index_str = os.environ.get("CLIENT_INDEX")
    if client_index_str is None:
        # Default to 0 if none
        logger.warning("No CLIENT_INDEX found, defaulting to 0 (benign).")
        client_index = 0
    else:
        client_index = int(client_index_str)

    logger.info(f"client_index={client_index}")

    param_path = '/var/parameter_store/param_store.json'
    if os.path.isfile(param_path):
        with open(param_path, 'r') as f:
            store = json.load(f)

        client_conf = next(
            (c for c in store.get("clients", []) if c["client_id"] == client_index),
            None
        )
        if client_conf:
            param_store_malicious = client_conf.get("is_malicious", False)
            malicious = env_malicious or param_store_malicious
            if malicious:
                attack = client_conf.get("attack_type", "none")
                inflation_factor = client_conf.get("inflation_factor", 1)
            else:
                attack = "none"
                inflation_factor = 1

            batch_size = store.get("batch_size", batch_size)
            epochs     = store.get("epochs", epochs)

        else:
            logger.warning(f"No client entry found for client_id={client_index}. Using defaults.")
            inflation_factor = 1
    else:
        logger.warning("No param_store.json found! Using all defaults.")
        inflation_factor = 1

    logger.info(f"[TRAIN] client_index={client_index}, malicious={malicious}, attack={attack}")
    logger.info(f"[TRAIN] hyperparams: epochs={epochs}, batch_size={batch_size}, inflation_factor={inflation_factor}")

    x_train, y_train = load_data(data_path)
    x_train = np.array(x_train)
    y_train = np.array(y_train)


    if malicious:
        logger.info(f"[DEBUG] Attack mode '{attack}' enabled.")
        match attack:
            case 'grad_boost_basic':
                logger.info(f"[Attack] grad_boost_basic: Coefs before boost: {model.coef_}")
                logger.info(f"[Attack] Applying boost factor {inflation_factor}")
                model.coef_ = inflation_factor * model.coef_
                model.intercept_ = inflation_factor * model.intercept_
                logger.info(f"[Attack] Coefs after boost: {model.coef_}")
            
            case 'label_flip_basic':
                L = 3 # number of classes in IRIS
                y_train = (L - 1 - y_train).tolist()  # convert back or keep as array
                print("[Attack] Label flipping attack done.")
            
            case None:
                logger.info("[Attack] Attack was set to None, ignoring.")

            case _:
                logger.warning(f"[Attack] Unrecognized attack: {attack}")


    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train)

    n_samples = len(x_train_df)
    classes = [0, 1, 2]

    rng = np.random.default_rng(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    x_train_df = x_train_df.iloc[indices].reset_index(drop=True)
    y_train_df = y_train_df.iloc[indices].reset_index(drop=True)

    if n_samples < batch_size:
        # If there's less data than one batch, feed all
        model.partial_fit(x_train_df, y_train_df.values.ravel(), classes=classes)
    else:
        x_batch = x_train_df.iloc[:batch_size]
        y_batch = y_train_df.iloc[:batch_size]
        model.partial_fit(x_batch, y_batch.values.ravel(), classes=classes)

    for e in range(epochs):
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        x_train_df = x_train_df.iloc[indices].reset_index(drop=True)
        y_train_df = y_train_df.iloc[indices].reset_index(drop=True)

        start_idx = 0
        while start_idx < n_samples:
            end_idx = start_idx + batch_size
            x_batch = x_train_df.iloc[start_idx:end_idx]
            y_batch = y_train_df.iloc[start_idx:end_idx]

            model.partial_fit(x_batch, y_batch.values.ravel(), classes=classes)

            start_idx += batch_size

        logger.info(f"[Epoch {e+1}/{epochs}] partial_fit complete.")



    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        'num_examples': len(x_train),
        'epochs': epochs,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)

    # #params_path = f"/var/parameters/{os.uname().nodename}"
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

if __name__ == "__main__":
    """
    Example usage:
    python train.py <in_model_path> <out_model_path> [<data_path> [<batch_size> <epochs> <malicious> <attack>]]
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

    if len(sys.argv) >= 4:
        batch_size = int(sys.argv[4])
    if len(sys.argv) >= 5:
        epochs = int(sys.argv[5])
    if len(sys.argv) >= 6:
        # "True" or "False"
        malicious_flag = (sys.argv[6].lower() == "true")
    if len(sys.argv) >= 7:
        attack_type = sys.argv[7]
    
    print('in_model', in_model, 'out_model', out_model, 'data_path', data_path, 'batch_size', batch_size, 'epochs', epochs, 'malicious_flag', malicious_flag, 'attack_type', attack_type)

    train(in_model, out_model, data_path, batch_size, epochs, malicious_flag, attack_type)
