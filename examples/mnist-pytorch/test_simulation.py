import subprocess
import yaml
from helper_simulation import helper
from fedn import APIClient
import time
import os

from server_functions_DNC import ServerFunctions as DNC
from server_functions_KRUM import ServerFunctions as KRUM
from server_functions_Multi_KRUM import ServerFunctions as Multi_KRUM
from server_functions_TrMean import ServerFunctions as TrMean
from server_functions_fedavg import ServerFunctions as FedAvg

def send_params_to_kubernetes_pods(helper_tuple):
    COMBINER_IP, CLIENT_TOKEN, ATTACK_TYPE, inflation_factor, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEFENSE_TYPE, BENIGN_CLIENTS, MALICIOUS_CLIENTS,DATA_ENDPOINT, DATA_ACCESS_KEY, DATA_SECRET_KEY, DATA_BUCKET_NAME, AUTH_TOKEN, client, IID, BALANCED = helper_tuple[0], helper_tuple[1], helper_tuple[2], helper_tuple[3], helper_tuple[4], helper_tuple[5], helper_tuple[6], helper_tuple[7], helper_tuple[8], helper_tuple[9], helper_tuple[10], helper_tuple[11], helper_tuple[12], helper_tuple[13], helper_tuple[14], helper_tuple[15], helper_tuple[16], helper_tuple[17]    
    
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
                "lr": LEARNING_RATE,
                "epochs": EPOCHS,
                "inflation_factor": inflation_factor,
                "data_endpoint": DATA_ENDPOINT,
                "data_access_key": DATA_ACCESS_KEY,
                "data_secret_key": DATA_SECRET_KEY,
                "data_bucket_name": DATA_BUCKET_NAME,
                "balanced": BALANCED,
                "iid": IID,
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
                "lr": LEARNING_RATE,
                "epochs": EPOCHS,
                "inflation_factor": inflation_factor,
                "data_endpoint": DATA_ENDPOINT,
                "data_access_key": DATA_ACCESS_KEY,
                "data_secret_key": DATA_SECRET_KEY,
                "data_bucket_name": DATA_BUCKET_NAME,
                "balanced": BALANCED,
                "iid": IID,
            }
        )

    with open("chart/values.yaml", "r") as f:
        values = yaml.safe_load(f)
    
    values["combinerIP"] = COMBINER_IP
    values["clientToken"] = CLIENT_TOKEN
    values["clients"] = client_list
    values["benign"]["replicas"] = BENIGN_CLIENTS
    values["malicious"]["replicas"] = MALICIOUS_CLIENTS
    values["authToken"] = AUTH_TOKEN

    with open("values-temp.yaml", "w") as f:
        yaml.safe_dump(values, f)
    
    helm_cmd = [
        "helm", "upgrade", "--install", "mnist-sim",
         "./chart", "-f", "values-temp.yaml"
    ]
    subprocess.run(helm_cmd, check=True)
    # print("Clients deployed with user-supplied config!")

    #time.sleep(5)
    print("Starting simulation...")

    if DEFENSE_TYPE == "dnc":
        ServerFunctions = DNC
    elif DEFENSE_TYPE == "krum":
        ServerFunctions = KRUM
    elif DEFENSE_TYPE == "multi-krum":
        ServerFunctions = Multi_KRUM
    elif DEFENSE_TYPE == "trmean":
        ServerFunctions = TrMean
    elif DEFENSE_TYPE == "fedavg":
        ServerFunctions = FedAvg
    session_name = input("Enter Session Name: ")
    
    #client.start_session(name=session_name, round_timeout=500, rounds=10)

    client.start_session(name=session_name, server_functions=ServerFunctions, round_timeout=500, rounds=10)

    #client.start_session(name=session_name, server_functions=ServerFunctions)
    #print("Simulation started!")


send_params_to_kubernetes_pods(helper())




