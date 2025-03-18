import os
import sys
import math
import torch
import collections
import json
import logging
from fedn.utils.helpers.helpers import save_metadata, get_helper
from data import load_data
from model import save_parameters, compile_model
from load_environment_param import load_env_params
from attacks import *

logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)

HELPER_MODULE = 'numpyhelper'
helper = get_helper(HELPER_MODULE)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

def train(model):
    """ Complete a model update.

    Load model paramters from model (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path and sends to docker_client.py.

    """
    # Read environment variables from Kubernetes pod
    client_index_str, malicious, attack, inflation_factor, batch_size, epochs, lr, _, _, _, _ = load_env_params()
    out_model_path = f"/app/model_update_{client_index_str}.npz"
    # Load data
    x_train, y_train = load_data(is_train=True)

    # Print for debugging
    logger.info(f"[TRAIN] client_index={client_index_str}, malicious={malicious}, attack={attack}")
    logger.info(f"[TRAIN] final hyperparams: epochs={epochs}, batch_size={batch_size}, lr={lr}, inflation_factor={inflation_factor}")

    # Implement different version of training for malicious clients
    if malicious:
        match attack:
            case 'label_flip_basic':
                y_train = label_flip(y_train)
            case 'backdoor_35int':
                x_train, y_train = backdoor_35int(x_train, y_train)
            case 'artificial_backdoor_05p':
                x_train, y_train = artificial_backdoor_05p(x_train, y_train)
            case 'artificial_backdoor_05p_center':
                x_train, y_train = artificial_backdoor_05p_center(x_train, y_train)
            case None:
                if attack == 'little_is_enough':
                    logger.info('LIE attack')
                else:
                    logger.warning('No attack was specified for the malicious client.')
            case _:
                logger.info("DO NOTHING!")

    if attack == 'little_is_enough':
        logger.info("This client is running a LIE attack!")
        pull_factor = 2
        with open('/var/parameter_store/client_counts.json', 'r') as json_file:
            counts = json.load(json_file)
            ben_count = counts['ben_count']
            mal_count = counts['mal_count']
            mal_power = int(ben_count / mal_count)
            logger.info(f"Malicious pull is {mal_power}")

        model_ids = [int(x.split(sep='.')[0]) for x in os.listdir("/var/parameter_store/models/")]
        model_count = len(model_ids)
        logger.info(f"Model count is: {model_count}")
        if (model_count != 0 and model_count % 3 == 0):
            latest_model_parameters_np = helper.load(f"/var/parameter_store/models/{model_ids[model_count - 1]}.npz")
            reference_model_parameters_np = helper.load(f"/var/parameter_store/models/{model_ids[model_count - 3]}.npz")
            updated_model_parameters_np = []

            for i in range(len(reference_model_parameters_np)):
                updated_model_parameters_np.append(reference_model_parameters_np[i] - pull_factor * mal_power * (latest_model_parameters_np[i] - reference_model_parameters_np[i]))
            
            model = compile_model()

            params_dict = zip(model.state_dict().keys(), updated_model_parameters_np)
            state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
            model.load_state_dict(state_dict, strict=True)
        else:
            logger.info(f"But not running the attack in this round since it is round no: {model_count}")
            # Train
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            n_batches = int(math.ceil(len(x_train) / batch_size))
            criterion = torch.nn.CrossEntropyLoss()
            for e in range(epochs):  # epoch loop
                for b in range(n_batches):  # batch loop
                    # Retrieve current batch
                    batch_x = x_train[b * batch_size:(b + 1) * batch_size]
                    batch_y = y_train[b * batch_size:(b + 1) * batch_size]

                    # Train on batch
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    # log/print loss
                    if b % 100 == 0:
                        logger.info(
                            f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")
    else:
        # Train
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        n_batches = int(math.ceil(len(x_train) / batch_size))
        criterion = torch.nn.CrossEntropyLoss()
        for e in range(epochs):  # epoch loop
            for b in range(n_batches):  # batch loop
                # Retrieve current batch
                batch_x = x_train[b * batch_size:(b + 1) * batch_size]
                batch_y = y_train[b * batch_size:(b + 1) * batch_size]
                # Train on batch
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                if malicious:
                    match attack:
                        case 'grad_boost_basic':
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad *= inflation_factor
                            
                optimizer.step()
                # log/print loss
                if b % 100 == 0:
                    logger.info(
                        f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

    # Metadata needed for aggregation server side
    metadata = {
        'num_examples': len(x_train),
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)

    logger.info('Train Completed!')
    return metadata, model
