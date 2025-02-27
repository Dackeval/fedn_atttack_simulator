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
    # Load data
    attack = 'none'
    malicious=False 
    x_train, y_train = load_data(data_path)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #model = load_parameters(in_model_path)
    #print('in_model', in_model_path, 'out_model', out_model_path, 'data_path', data_path, 'batch_size', batch_size, 'epochs', epochs, 'malicious_flag', malicious_flag, 'attack_type', attack_type)


    if malicious:
        print(f"[DEBUG] Attack mode '{attack}' enabled.")
        match attack:
            case 'grad_boost_basic':
                print(f"[Attack] grad_boost_basic: Coefs before boost: {model.coef_}")
                inflation_factor = 100
                print(f"[Attack] A boost factor of {inflation_factor} is applied on the parameters!")
                model.coef_ = inflation_factor * model.coef_
                model.intercept_ = inflation_factor * model.intercept_
                print(f"[Attack] Coefs after boost: {model.coef_}")

            case 'label_flip_basic':
                L = 3 # number of classes in IRIS
                y_train = (L - 1 - y_train).tolist()  # convert back or keep as array
                print("[Attack] Label flipping attack done.")

            case None:
                print("[Attack] Attack was set to None, ignoring.")

            case _:
                print(f"[Attack] Unrecognized attack: {attack}")


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

        print(f"[Epoch {e+1}/{epochs}] partial_fit complete.")

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
    
    print('Train Completed!')
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
