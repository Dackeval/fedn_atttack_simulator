import subprocess
import yaml
from fedn import APIClient
import time
import os
from pathlib import Path

from server_functions import ServerFunctions
from bin.split_data import split as sd



def load_config(path_to_yaml: str | None = None):
    """
    Load the YAML config. Priority:
    1. explicit path passed in
    2. ./config.yaml next to this script
    """
    if path_to_yaml is None:
        # /path/to/examples/mnist-pytorch/config.yaml
        path_to_yaml = Path(__file__).with_name("config.yaml")

    with open(path_to_yaml, "r") as fh:
        return yaml.safe_load(fh)

def send_params_to_kubernetes_pods():
    #COMBINER_IP, CLIENT_TOKEN, ATTACK_TYPE, inflation_factor, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEFENSE_TYPE, BENIGN_CLIENTS, MALICIOUS_CLIENTS,DATA_ENDPOINT, DATA_ACCESS_KEY, DATA_SECRET_KEY, DATA_BUCKET_NAME, AUTH_TOKEN, client, IID, BALANCED = helper_tuple[0], helper_tuple[1], helper_tuple[2], helper_tuple[3], helper_tuple[4], helper_tuple[5], helper_tuple[6], helper_tuple[7], helper_tuple[8], helper_tuple[9], helper_tuple[10], helper_tuple[11], helper_tuple[12], helper_tuple[13], helper_tuple[14], helper_tuple[15], helper_tuple[16], helper_tuple[17]    
    print("Welcome to the attack simulation!")


    config = load_config()
    sim_cfg = config["simulation"]

    # Data splits 
    pushfetch_or_fetch = input("Do you want to split, push and fetch data partitions or only fetch the data partitions remotely? (push/fetch): ")
    if pushfetch_or_fetch == "push":
        sd(sim_cfg["benign_clients"] + sim_cfg["malicious_clients"], sim_cfg["data_endpoint"], sim_cfg["data_access_key"], sim_cfg["data_secret_key"], sim_cfg["data_bucket_name"], sim_cfg["iid"], sim_cfg["balanced"])


    client_list = []    
    # Create a list of clients with their parameters
    # The first clients are benign clients
    for i in range(sim_cfg["benign_clients"]):
        client_index = i + 1
        client_list.append(
            {
                "id": client_index,
                "is_malicious": False,
                "attack_type": sim_cfg["attack_type"],
                "batch_size": sim_cfg["batch_size"],
                "lr": sim_cfg["learning_rate"],
                "epochs": sim_cfg["epochs"],
                "inflation_factor": sim_cfg["inflation_factor"],
                "data_endpoint": sim_cfg["data_endpoint"],
                "data_access_key": sim_cfg["data_access_key"],
                "data_secret_key": sim_cfg["data_secret_key"],
                "data_bucket_name": sim_cfg["data_bucket_name"],
                "balanced": sim_cfg["balanced"],
                "iid": sim_cfg["iid"],
            }
        )
    # The malicious clients are numbered from BENIGN_CLIENTS + 1 to total_clients
    for i in range(sim_cfg["malicious_clients"]):
        client_index = sim_cfg["benign_clients"] + i + 1
        client_list.append(
            {
                "id": client_index,
                "is_malicious": True,
                "attack_type": sim_cfg["attack_type"],
                "batch_size": sim_cfg["batch_size"],
                "lr": sim_cfg["learning_rate"],
                "epochs": sim_cfg["epochs"],
                "inflation_factor": sim_cfg["inflation_factor"],
                "data_endpoint": sim_cfg["data_endpoint"],
                "data_access_key": sim_cfg["data_access_key"],
                "data_secret_key": sim_cfg["data_secret_key"],
                "data_bucket_name": sim_cfg["data_bucket_name"],
                "balanced": sim_cfg["balanced"],
                "iid": sim_cfg["iid"],
            }
        )
    # Create a list of clients with their parameters
    with open("chart/values.yaml", "r") as f:
        values = yaml.safe_load(f)
    # non client specific values
    values["combinerIP"] = sim_cfg["combiner_ip"]
    values["clientToken"] = sim_cfg["client_token"]
    values["clients"] = client_list # dump the client list to the values.yaml file
    values["benign"]["replicas"] = sim_cfg["benign_clients"]
    values["malicious"]["replicas"] = sim_cfg["malicious_clients"]
    values["authToken"] = sim_cfg["auth_token"]
    values["defense_type"] = sim_cfg["defense_type"]
    values["late_client_ind"] = sim_cfg["late_client_ind"]
    values["late_client_delay"] = sim_cfg["late_client_delay"]
    # dump the values to a temporary file
    with open("values-temp.yaml", "w") as f:
        yaml.safe_dump(values, f)
    # run helm command to deploy the chart
    print("Deploying chart...")
    # run helm with the temporary values file
    helm_cmd = [
        "helm", "upgrade", "--install", "mnist-sim",
         "./chart", "-f", "values-temp.yaml"
    ]
    subprocess.run(helm_cmd, check=True)
    # wait for the pods to be ready
    print("Waiting for pods to be ready...")
    time.sleep(20)

    session_name = input("Enter Session Name: ")
    os.environ["FEDN_AUTH_TOKEN"] = sim_cfg["auth_token"]
    discover_host = sim_cfg["combiner_ip"].removeprefix('https://')
    client = APIClient(host=discover_host, secure=True, verify=True)
    
    print(client.start_session(
        name=session_name,
        server_functions=ServerFunctions,
        round_timeout=300,
        rounds=sim_cfg["rounds"]))
    print("Simulation started!")


send_params_to_kubernetes_pods()




