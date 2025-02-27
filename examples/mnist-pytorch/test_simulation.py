from fedn import APIClient
import time
import docker
import sys
import subprocess
import json
# sys.path.append('/home/ubuntu/fedn-attack-sim-uu/examples/mnist-pytorch')
import os
import bin.parameter_store as ps
from bin.split_data import split as sd

def start_clients(combiner_ip, token, benign_client_count, malicious_client_count):
    script_path = './bin/start_clients.sh'  # Path to your shell script
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

def get_combiner_ip():
    with open("/home/ubuntu/fedn-attack-sim-uu/simulator/config/api_server_config.json", "r") as file:
        config = json.load(file)
    
    if config['initialized']:
        return config['api_server_ip']
    else:
        print(f"API configuration has not been set!")
        exit()

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


COMBINER_IP = input("Enter host IP: ")
print(f"Combiner IP: {COMBINER_IP} is set\n")

TOKEN = input("Enter token: ")
print(f"Token: {TOKEN} is set\n")

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

DEFENSE_TYPE = input("Enter defense type: ")
print(f"Defense type: {DEFENSE_TYPE} is set\n")

BENIGN_CLIENTS = get_valid_int("Enter number of benign clients (integer): ")
print(f"Benign clients: {BENIGN_CLIENTS} is set\n")

MALICIOUS_CLIENTS = get_valid_int("Enter number of malicious clients (integer): ")
print(f"Malicious clients: {MALICIOUS_CLIENTS} is set\n")


# check if ip and token works
DISCOVER_HOST = COMBINER_IP

try:
    os.environ["FEDN_AUTH_TOKEN"] = TOKEN
    client = APIClient(host=COMBINER_IP, secure=True, verify=True)
    print(f"API Client connected to combiner at: {DISCOVER_HOST}")
except Exception as e:
    print(f"Error connecting to combiner: {e}")
    COMBINER_IP = input("Enter host IP: ")
    print(f"Combiner IP: {COMBINER_IP} is set")
    TOKEN = input("Enter token: ")
    print(f"Token: {TOKEN} is set")
    os.environ["FEDN_AUTH_TOKEN"] = TOKEN
    client = APIClient(host=COMBINER_IP, secure=True, verify=True)
    print(f"API Client connected to combiner at: {DISCOVER_HOST}")

# ------------------------------
# NEED TO ADD IDENTIFICATION OF BENIGN AND MALICIOUS CLIENTS TO THE PARAMETER STORE JSON FILE
# Maybe id by number, last is malicious.. 
# ------------------------------
# Write the parameters to the parameter store
ps.create_parameter_store(BENIGN_CLIENTS, MALICIOUS_CLIENTS, ATTACK_TYPE, DEFENSE_TYPE, COMBINER_IP, TOKEN, LEARNING_RATE, EPOCHS, BATCH_SIZE, inflation_factor)

# ------------------------------
# SPLIT DATA
total_clients = BENIGN_CLIENTS + MALICIOUS_CLIENTS
sd(total_clients)
# ------------------------------

# ------------------------------
# ( NEED TO RUN BUILD SCRIPT TO CREATE THE PACKAGE AND SEED MODEL )
# ------------------------------
# # Upload seed model and package
# client.set_active_model('./client/seed.npz')
# package_name = str(input("Enter package name: "))
# client.set_active_package('./client/package.tgz', 'numpyhelper', package_name)




# CLIENTSs
docker_client = docker.from_env()
running_containers = docker_client.containers.list()

if len(running_containers) != 0:
    print(f"{len(running_containers)} clients are running!")
    for id, container in enumerate(running_containers):
        print(f"{id} - {container.name}")
else:
    print("No containers are running!")
    start_clients(COMBINER_IP, TOKEN, BENIGN_CLIENTS, MALICIOUS_CLIENTS)




# ------------------------------
# SET ACTIVE PACKAGE AND MODEL
# START SESSION
# RUN UNTIL FINISHED
# KILL CLIENTS
# ------------------------------


# client.set_active_package('package.tgz', 'numpyhelper')
# client.set_active_model('seed.npz')
# seed_model = client.get_initial_model()


# session_id = input("Set session_id: ")
# rounds = input("Set the number of rounds: ")

# session_config_fedavg = {
#     "helper": "numpyhelper",
#     "session_id": session_id,
#     "aggregator": "fedavg",
#     "model_id": seed_model['model_id'],
#     "rounds": int(rounds)
# }

# result_fedavg = client.start_session(**session_config_fedavg)
# time.sleep(10)

# def run_until_finished(session_id):
#     while not client.session_is_finished(session_id):
#         models = client.list_models(session_id)
#         print(f"Rounds: {models['count']} out of {session_config_fedavg['rounds']} completed!", end="\r")
#         time.sleep(15)

# # Call the function
# run_until_finished(session_id) 

# if client.session_is_finished(session_id):
#     print(f"The session: {session_id} is over!")
#     kill_clients()