import os
import sys
import math
import torch
import collections
import json

from fedn.utils.helpers.helpers import save_metadata, get_helper
from data import load_data
from model import load_parameters, save_parameters, compile_model

HELPER_MODULE = 'numpyhelper'
helper = get_helper(HELPER_MODULE)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

def train(
    model,
    out_model_path='/app/model_update.npz',
    data_path=None,
    batch_size=32,
    epochs=1,
    lr=0.01,
    malicious=False,
    attack='none'
):
    """ Complete a model update.s

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

    x_train, y_train = load_data(data_path)


    # Check if container sets MALICIOUS=true
    env_malicious_flag = os.environ.get("MALICIOUS", "false").strip().lower()
    # Convert to boolean
    env_malicious = (env_malicious_flag == "true")


    client_index_str = os.environ.get("CLIENT_INDEX")
    if client_index_str is None:
        # Default to 0 if none
        print("No CLIENT_INDEX found, defaulting to 0 (benign).")
        client_index = 0
    else:
        client_index = int(client_index_str)


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
            lr         = store.get("lr", lr)

        else:
            print(f"No client entry found for client_id={client_index}. Using defaults.")

            inflation_factor = 1
    else:
        print("No param_store.json found! Using all defaults.")
        inflation_factor = 1

    print(f"[TRAIN] client_index={client_index}, malicious={malicious}, attack={attack}")
    print(f"[TRAIN] final hyperparams: epochs={epochs}, batch_size={batch_size}, lr={lr}, inflation_factor={inflation_factor}")

    # Implement different version of training for malicious clients
    if malicious:
        print('debug 2')
        match attack:
            case 'grad_boost_basic':
                ### Gradient inflation attack ###
                # DO NOTHING IN THE DATA LOADING PROCESS
                print("DO NOTHING IN THE DATA LOADING PROCESS")
                ### End of inflation attack code ###
            case 'label_flip_basic':
                ### Label flipping attack - basic
                y_train_unflipped = y_train
                L = len(set([value.item() for value in y_train]))
                y_train = torch.tensor([L - 1 - y for y in y_train])
                print('Running a label flip :D')
                ### End of label flipping attack - basic
            case 'backdoor_35int':
                # Implementation of the backdoor attack
                target_label = 8 # Label which we try to misclassify as the backdoor label
                print(f"[Attack Training]: Running a backdoor attack to misclassify label {target_label}")

                # Inject backdoor to backdoor label
                for index, is_target in enumerate((y_train == target_label).tolist()):
                    if is_target:
                        # Backdoor intensity is defined here
                        x_train[index][1] = torch.tensor([0.9922 for x in range(28)])

                prop_counter = 0
                bd_prop = 0.2

                for backdoor_label in range(10):
                    for index, is_target in enumerate((y_train == backdoor_label).tolist()):
                        if is_target:
                            if prop_counter == 0:
                                x_train[index][1] = torch.tensor([0.9922 for x in range(28)])
                                prop_counter += 1
                            else:
                                if prop_counter < int(int(1 / bd_prop)):
                                    prop_counter += 1
                                else:
                                    prop_counter = 0
            case 'artificial_backdoor_05p':
                backdoor_label = 8
                intensity = 1
                print(f"Adding a backdoor trigger to label {backdoor_label}")
                for index, is_target in enumerate((y_train == backdoor_label).tolist()):
                    if is_target:
                        x_train[index][2] = torch.tensor([intensity if (x > 4 and x <= 5) else 0 for x in range(28)])
                        x_train[index][3] = torch.tensor([intensity if (x > 3 and x <= 6) else 0 for x in range(28)])
                        x_train[index][4] = torch.tensor([intensity if (x > 4 and x <= 5) else 0 for x in range(28)])
            case 'artificial_backdoor_05p_center':
                backdoor_label = 8
                print(f"Adding a backdoor trigger to label {backdoor_label}")
                for index, is_target in enumerate((y_train == backdoor_label).tolist()):
                    if is_target:
                        x_train[index][6][8] = 1
                        x_train[index][7][7] = 1
                        x_train[index][7][8] = 1
                        x_train[index][7][9] = 1
                        x_train[index][8][8] = 1
            case None:
                if attack == 'little_is_enough':
                    print('LIE attack')
                else:
                    print('No attack was specified for the malicious client.')
            case _:
                print("DO NOTHING!")

    # Load parmeters and initialize model
    #model = load_parameters(in_model_path)

    if attack == 'little_is_enough':
        print("This client is running a LIE attack!")
        pull_factor = 2
        with open('/var/parameter_store/client_counts.json', 'r') as json_file:
            counts = json.load(json_file)
            ben_count = counts['ben_count']
            mal_count = counts['mal_count']
            mal_power = int(ben_count / mal_count)
            print(f"Malicious pull is {mal_power}")

        model_ids = [int(x.split(sep='.')[0]) for x in os.listdir("/var/parameter_store/models/")]
        model_count = len(model_ids)
        print(f"Model count is: {model_count}")
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
            print(f"But not running the attack in this round since it is round no: {model_count}")
            # Train
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            n_batches = int(math.ceil(len(x_train) / batch_size))
            criterion = torch.nn.NLLLoss()
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

                    # Implement different version of training for malicious clients
                    if malicious:
                        match attack:
                            case 'grad_boost_basic':
                                ### Gradient inflation attack ###
                                inflation_factor = 2  # Can be adjusted
                                print(f"An inflation factor of {inflation_factor} is applied on the parameters!")
                                for param in model.parameters():
                                    if param.grad is not None:
                                        param.grad *= inflation_factor
                                ### End of inflation attack code ###
                            case 'label_flip_basic':
                                ### Label flipping attack - basic
                                print("DO NOTHING IN THE TRAINING PROCESS")
                                ### End of label flipping attack - basic
                            case 'little_is_enough':
                                print(f"The attack specified was LIE, but the model is not attacked in this round since the model count is {model_count}")
                            case None:
                                print('No attack was specified for the malicious client.')
                                
                    optimizer.step()
                    # Log
                    if b % 100 == 0:
                        print(
                            f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")
    else:
        # Train
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        n_batches = int(math.ceil(len(x_train) / batch_size))
        criterion = torch.nn.NLLLoss()
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

                # Implement different version of training for malicious clients
                if malicious:
                    match attack:
                        case 'grad_boost_basic':
                            ### Gradient inflation attack ###
                            inflation_factor = 100  # Can be adjusted
                            print(f"An inflation factor of {inflation_factor} is applied on the parameters!")
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad *= inflation_factor
                            ### End of inflation attack code ###
                        case 'label_flip_basic':
                            ### Label flipping attack - basic
                            print("DO NOTHING IN THE TRAINING PROCESS")
                            ### End of label flipping attack - basic
                        case None:
                            print('No attack was specified for the malicious client.')
                            
                optimizer.step()
                # Log
                if b % 100 == 0:
                    print(
                        f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        'num_examples': len(x_train),
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)
    print('Train Completed!')
    return metadata, model



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
    batch_size = 32
    epochs = 1
    lr = 0.01
    malicious_flag = False
    attack_type = None

    if len(sys.argv) > 4:
        batch_size = int(sys.argv[4])
    if len(sys.argv) > 5:
        epochs = int(sys.argv[5])
    if len(sys.argv) > 6:
        lr = float(sys.argv[6])
    if len(sys.argv) > 7:
        # "True" or "False"
        malicious_flag = (sys.argv[7].lower() == "true")
    if len(sys.argv) > 8:
        attack_type = sys.argv[8]
    
    print('in_model:', in_model, 'out_model:', out_model, 'data_path:', data_path, 'batch_size:', batch_size, 'epochs:', epochs, 'lr:', lr, 'malicious:', malicious_flag, 'attack:', attack_type)

    train(in_model, out_model, data_path, batch_size, epochs, lr, malicious_flag, attack_type)
