# For User inputted parameters for the simulation, currently not used, but config.yaml is default.


from fedn import APIClient
import os
from bin.split_data import split as sd
from server_functions_fedavg import ServerFunctions

def get_valid_int(prompt):
    """
    prompt for an int until valid input
    """
    while True:
        value_str = input(prompt)
        try:
            value_int = int(value_str)
            return value_int
        except ValueError:
            print("Invalid input. Please enter a valid integer.\n")

def get_valid_float(prompt):
    """
    prompt for an float until valid input
    """
    while True:
        value_str = input(prompt)
        try:
            value_float = float(value_str)
            return value_float
        except ValueError:
            print("Invalid input. Please enter a valid float.\n")


def get_valid_attack_type(prompt):
    """
    prompt for a valid attack type until valid
    """
    valid_types = [
        "label_flip_basic",
        "grad_boost_basic",
        "little_is_enough",
        "artificial_backdoor_05p_center",
        "artificial_backdoor_05p",
        "backdoor_35int"
    ]
    while True:
        attack_type = input(prompt).strip()
        if attack_type in valid_types:
            return attack_type
        else:
            print("Invalid attack type. Please choose from:")
            for t in valid_types:
                print("  -", t)
            print()

def get_valid_mitigation_type(prompt):
    """
    prompt for a valid attack type until valid
    """
    valid_types = [
        "dnc",
        "krum",
        "multi-krum",
        "trmean",
        "exploration-exploitation",
        "fedavg"
    ]
    while True:
        mitigation_type = input(prompt).strip()
        if mitigation_type in valid_types:
            return mitigation_type
        else:
            print("Invalid attack type. Please choose from:")
            for t in valid_types:
                print("  -", t)
            print()


def helper():
    """
    helper function to get the parameters for the simulation
    """
    print("Welcome to the Attack Simulator!\n")

    # simulator parameter inputs
    # -----------------------------------------------------
    COMBINER_IP = input("Enter host IP: ")
    print(f"Combiner IP: {COMBINER_IP} is set\n")

    CLIENT_TOKEN = input("Enter client token: ")
    print(f"Client token: {CLIENT_TOKEN} is set\n")

    ATTACK_TYPE = get_valid_attack_type("Enter attack type: ")
    print(f"Attack type: {ATTACK_TYPE} is set\n")

    inflation_factor = 1
    if ATTACK_TYPE == "grad_boost_basic":
        inflation_factor = get_valid_int("Enter inflation factor (integer): ")
        print(f"Inflation factor: {inflation_factor} is set\n")

    BATCH_SIZE = get_valid_int("Enter batch size (integer): ")
    print(f"Batch size: {BATCH_SIZE} is set\n")

    EPOCHS = get_valid_int("Enter number of epochs (integer): ")
    print(f"Epochs: {EPOCHS} is set\n")

    LEARNING_RATE = get_valid_float("Enter learning rate (float): ")
    print(f"Learning rate: {LEARNING_RATE} is set\n")

    DEFENSE_TYPE = get_valid_mitigation_type("Enter defense type: ")
    print(f"Defense type: {DEFENSE_TYPE} is set\n")

    BENIGN_CLIENTS = get_valid_int("Enter number of benign clients (integer): ")
    print(f"Benign clients: {BENIGN_CLIENTS} is set\n")

    MALICIOUS_CLIENTS = get_valid_int("Enter number of malicious clients (integer): ")
    print(f"Malicious clients: {MALICIOUS_CLIENTS} is set\n")

    DATA_ENDPOINT = input("Enter the data endpoint: ")
    print(f"Data_endpoint is set: '{DATA_ENDPOINT}'")

    DATA_ACCESS_KEY = input("Enter the data access key: ")
    print(f"Data access key is set: '{DATA_ACCESS_KEY}'")

    DATA_SECRET_KEY = input("Enter the data secret key: ")
    print(f"Data secret key is set: '{DATA_SECRET_KEY}'")

    DATA_BUCKET_NAME = input("Enter the data bucket name: ")
    print(f"Data bucket name is set: '{DATA_BUCKET_NAME}'")

    IID = input("Is the data IID? (y/n): ")
    if IID == "y":
        IID = "iid"
    elif IID == "n":
        IID = "noniid"
    else:
        print("Invalid input. Defaulting to IID.")
        IID = "iid"
    
    BALANCED = input("Is the data balanced? (y/n): ")
    if BALANCED == "y":
        BALANCED = "balanced"
    elif BALANCED == "n":
        BALANCED = "unbalanced"
    else:
        print("Invalid input. Defaulting to balanced.")
        BALANCED = "balanced"
    # -----------------------------------------------------
    
    # split data
    total_clients = BENIGN_CLIENTS + MALICIOUS_CLIENTS
    pushfetch_or_fetch = input("Do you want to split, push and fetch data partitions or only fetch the data partitions remotely? (push/fetch): ")
    if pushfetch_or_fetch == "push":
        sd(total_clients, DATA_ENDPOINT, DATA_ACCESS_KEY, DATA_SECRET_KEY, DATA_BUCKET_NAME, IID, BALANCED)

    # old code, fetches token and client for downstream tasks 
    AUTH_TOKEN, client = send_seed_and_package(COMBINER_IP)

    return (COMBINER_IP, CLIENT_TOKEN, ATTACK_TYPE, inflation_factor, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEFENSE_TYPE, BENIGN_CLIENTS, MALICIOUS_CLIENTS, DATA_ENDPOINT, DATA_ACCESS_KEY, DATA_SECRET_KEY, DATA_BUCKET_NAME, AUTH_TOKEN, client, IID, BALANCED)


def send_seed_and_package(COMBINER_IP):
    DISCOVER_HOST = COMBINER_IP.removeprefix('https://')
    client_path = os.path.join(os.path.dirname(__file__), 'client')

    auth_token = str(input("Enter auth_token: "))

    os.environ["FEDN_AUTH_TOKEN"] = auth_token
    client = APIClient(host=DISCOVER_HOST, secure=True, verify=True)

    return auth_token, client


    