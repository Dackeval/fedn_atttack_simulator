from fedn import APIClient
import time
import docker
import sys
import subprocess
import json
import os
import bin.parameter_store as ps
from bin.split_data import split as sd
import yaml


def start_clients(combiner_ip, token, benign_client_count, malicious_client_count):
    script_path = '/Users/sigvard/Desktop/fedn_attack_simulator/examples/iris-sklearn/bin/start_clients.sh'  
    client = docker.from_env()

    try:
        # Run the shell script with the provided arguments
        print("running the shell script")
        result = subprocess.run(f"{script_path} {combiner_ip} {token} {int(benign_client_count)} {int(malicious_client_count)}", shell=True, check=True, text=True, capture_output=True)
        print(f"{len(client.containers.list())} clients started!")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the script: {e}")
        print(e.output)

def kill_clients():
    client = docker.from_env()
    for container in client.containers.list():
        try:
            print(f"Killing container: {container.name} ({container.id})")
            container.kill()
            print(f"Removing container: {container.name} ({container.id})")
            container.remove()
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing the script: {e}")
            print(e.output)
    
    if len(client.containers.list()) == 0:
        print("All running clients have been killed.")
    else:
        print(f"{len(client.containers.list())} clients are still running")

    # Delete split data
    try:
        # HARDCODED
        result = subprocess.run('sudo rm -rf data/clients/', shell=True, check=True, text=True, capture_output=True)
        print(f"All split data has been deleted!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the script: {e}")
        print(e.output)

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

def create_seed_package(cmd, cwd=None):
    activate_env = "source ../../fedn_env/bin/activate && "
    full_cmd = activate_env + cmd
    res = subprocess.run(full_cmd, shell=True, cwd=cwd, capture_output=True, text=True, executable='/bin/bash')
    if res.returncode != 0:
        print(f"Command '{cmd}' failed with error:\n{res.stderr}")
    else:
        print(f"Command '{cmd}' succeeded:\n{res.stdout}")

def get_valid_attack_type(prompt):
    """
    prompt for a valid attack type until valid
    """
    valid_types = [
        "label_flip_basic",
        "grad_boost_basic"
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

def send_params_to_kubernetes_pods(COMBINER_IP, CLIENT_TOKEN, 
                                   ATTACK_TYPE, inflation_factor,
                                   BATCH_SIZE,
                                   DEFENSE_TYPE, BENIGN_CLIENTS, MALICIOUS_CLIENTS,
                                   DATA_ENDPOINT, DATA_ACCESS_KEY, 
                                   DATA_SECRET_KEY, DATA_BUCKET_NAME
                                   ):
    total_clients = BENIGN_CLIENTS + MALICIOUS_CLIENTS
    client_list = []

    for i in range(BENIGN_CLIENTS):
        client_index = i + 1
        client_list.append(
            {
                "id": client_index,
                "is_malicious": False,
                "attack_type": ATTACK_TYPE,
                "batch_size": BATCH_SIZE,
                "inflation_factor": inflation_factor,
                "data_endpoint": DATA_ENDPOINT,
                "data_access_key": DATA_ACCESS_KEY,
                "data_secret_key": DATA_SECRET_KEY,
                "data_bucket_name": DATA_BUCKET_NAME
            }
        )
    for i in range(MALICIOUS_CLIENTS):
        client_index = BENIGN_CLIENTS + i + 1
        client_list.append(
            {
                "id": client_index,
                "is_malicious": True,
                "attack_type": ATTACK_TYPE,
                "batch_size": BATCH_SIZE,
                "inflation_factor": inflation_factor,
                "data_endpoint": DATA_ENDPOINT,
                "data_access_key": DATA_ACCESS_KEY,
                "data_secret_key": DATA_SECRET_KEY,
                "data_bucket_name": DATA_BUCKET_NAME
            }
        )

    with open("chart/values.yaml", "r") as f:
        values = yaml.safe_load(f)
    
    values["combinerIP"] = COMBINER_IP
    values["clientToken"] = CLIENT_TOKEN
    values["clients"] = client_list
    values["benign"]["replicas"] = BENIGN_CLIENTS
    values["malicious"]["replicas"] = MALICIOUS_CLIENTS

    with open("values-temp.yaml", "w") as f:
        yaml.safe_dump(values, f)
    
    helm_cmd = [
        "helm", "upgrade", "--install", "iris-sim",
         "./chart", "-f", "values-temp.yaml"
    ]
    subprocess.run(helm_cmd, check=True)
    print("Clients deployed with user-supplied config!")


# Simulator parameter inputs
# ----------------------------
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

DEFENSE_TYPE = input("Enter defense type: ")
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

# Write the parameters to the parameter store
# ------------------------------
#ps.create_parameter_store(BENIGN_CLIENTS, MALICIOUS_CLIENTS, ATTACK_TYPE, DEFENSE_TYPE, COMBINER_IP, CLIENT_TOKEN, EPOCHS, BATCH_SIZE, inflation_factor)

# SPLIT DATA
# ------------------------------
total_clients = BENIGN_CLIENTS + MALICIOUS_CLIENTS
#sd(total_clients, DATA_ENDPOINT, DATA_ACCESS_KEY, DATA_SECRET_KEY, DATA_BUCKET_NAME)

# UPLOAD PACKAGE AND SEED MODEL
# ------------------------------
DISCOVER_HOST = COMBINER_IP.removeprefix('https://')
client_path = os.path.join(os.path.dirname(__file__), 'client')
project_path = os.path.dirname(__file__)
seed_model_path = os.path.join(client_path, 'seed.npz')

try:
    auth_token = str(input("Enter auth_token: "))

    os.environ["FEDN_AUTH_TOKEN"] = auth_token
    client = APIClient(host=DISCOVER_HOST, secure=True, verify=True)

    # Upload seed model and package
    client.set_active_model(seed_model_path)
    package_name = str(input('Enter package name: '))
    client.set_active_package('./client/package.tgz', 'numpyhelper', package_name)

    print(f"API Client connected to combiner at: {DISCOVER_HOST}")
except Exception as e:
    print(f"Error connecting to combiner: {e}")

    COMBINER_IP = input("Enter host IP: ")
    print(f"Combiner IP: {DISCOVER_HOST} is set")
    auth_token = str(input("Enter auth_token: "))
    print(f"Token: {auth_token} is set")

    os.environ["FEDN_AUTH_TOKEN"] = auth_token
    client = APIClient(host=COMBINER_IP, secure=True, verify=True)

    client.set_active_model(seed_model_path)
    client.set_active_package('./client/package.tgz', 'numpyhelper', package_name)
    print(f"API Client connected to combiner at: {DISCOVER_HOST}")


send_params_to_kubernetes_pods(
  COMBINER_IP, CLIENT_TOKEN, ATTACK_TYPE, inflation_factor,
  BATCH_SIZE, DEFENSE_TYPE,
  BENIGN_CLIENTS, MALICIOUS_CLIENTS, DATA_ENDPOINT, 
  DATA_ACCESS_KEY, DATA_SECRET_KEY, DATA_BUCKET_NAME)

# # CLIENTSs
# docker_client = docker.from_env()
# running_containers = docker_client.containers.list()

# if len(running_containers) != 0:
#     print(f"{len(running_containers)} clients are running!")
#     for id, container in enumerate(running_containers):
#         print(f"{id} - {container.name}")
# else:
#     print("No containers are running!")
#     start_clients(COMBINER_IP, CLIENT_TOKEN, BENIGN_CLIENTS, MALICIOUS_CLIENTS)



# If we want to launch sessions from the simulator! 
# --------------------------------------------------
# time.sleep(10)


# session_name = str(input("Set session_id: "))
# rounds = int(input("Set the number of rounds: "))
# aggr = str(input("Set aggregator: "))
# active_model = client.get_active_model()
# model_id = active_model['id']
# round_timeout = int(input("Set round timeout: "))
# rounds = int(input('Set number of rounds: '))


# session_config_fedavg = {
#     "name": session_name,
#     "aggregator": aggr,
#     "model_id": model_id,
#     "round_timeout": round_timeout,
#     "rounds": rounds,
#     "round_buffer_size": int(-1),
#     "delete_models": True,
#     "validate": True,
#     "helper": "numpyhelper",
#     "min_clients": int(1),
#     "requested_clients": int(8)
# }

# result_fedavg = client.start_session(**session_config_fedavg)
# time.sleep(10)

# def run_until_finished(session_name):
#     while not client.session_is_finished(session_name):
#         models = client.get_models(session_name)
#         print(f"Rounds: {models['count']} out of {session_config_fedavg['rounds']} completed!", end="\r")
#         time.sleep(15)

# # # Call the function
# run_until_finished(session_name) 

# if client.session_is_finished(session_name):
#     print(f"The session: {session_name} is over!")
#     kill_clients()
# --------------------------------------------------