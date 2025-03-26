import os
import sys
import math
import torch
import collections
import json
import logging
from fedn.utils.helpers.helpers import save_metadata, get_helper
from data import load_data
from model import save_parameters, compile_model, load_parameters
from load_environment_param import load_env_params
from attacks import *
from fedn import APIClient


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
    # to keep track of the round number for LIE attack
    round_num = fetch_latest_round()
    logger.info(f"Round number is {round_num}")
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

    if attack == 'little_is_enough' and malicious:
        logger.info("This client is running a LIE attack!")
        pull_factor = 2
        ben_count = int(os.environ.get('BENIGN_CLIENTS', 0))
        mal_count = int(os.environ.get('MALICIOUS_CLIENTS', 0))
        if mal_count > 0:
            mal_power = int(ben_count / mal_count)
        else:
            mal_power = 1
        logger.info(f"Malicious pull is {mal_power}")
        if (round_num >= 1 and ( round_num - 2 ) % 3 == 0 ):
            save_parameters(model, "/app/model_t-2.npz")
            logger.info("Saved the model from 3 rounds ago")

        if (round_num != 0 and round_num % 3 == 0):
            # load the global model from 3 rounds ago
            global_model_parameters_t_2 = helper.load("/app/model_t-2.npz")
            global_model_parameters_t = [val.cpu().numpy() for _, val in model.state_dict().items()]
            
            updated_model_parameters_np = []
            for i in range(len(global_model_parameters_t)):
                updated_model_parameters_np.append(global_model_parameters_t[i] - pull_factor * mal_power * (global_model_parameters_t_2[i] - global_model_parameters_t[i]))
            
            model = compile_model()
            params_dict = zip(model.state_dict().keys(), updated_model_parameters_np)
            state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
            model.load_state_dict(state_dict, strict=True)

        else:
            logger.info(f"But not running the attack in this round since it is round no: {round_num}")
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


def fetch_latest_round():
    """Fetch the latest round count using the FEDn APIClient."""
    COMBINER_IP = str(os.getenv("COMBINER_IP", ""))
    DISCOVER_HOST = COMBINER_IP.removeprefix('https://')

    # # Get authentication token
    AUTH_TOKEN = str(os.getenv("AUTH_TOKEN", ""))
    os.environ["FEDN_AUTH_TOKEN"] = AUTH_TOKEN

    logger.info(f"COMBINER_IP: {COMBINER_IP}")
    logger.info(f"AUTH_TOKEN: {AUTH_TOKEN}")
    # Initialize API client
    client = APIClient(host=DISCOVER_HOST, secure=True, verify=True)

    # Fetch round count
    rounds_count = client.get_rounds_count()
    logger.info(f"Total Rounds: {rounds_count}")

    return int(rounds_count)
