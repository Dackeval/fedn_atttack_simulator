import subprocess
import yaml
from helper_simulation import helper


def send_params_to_kubernetes_pods(helper_tuple):
    COMBINER_IP, CLIENT_TOKEN, ATTACK_TYPE, inflation_factor,BATCH_SIZE, DEFENSE_TYPE, BENIGN_CLIENTS, MALICIOUS_CLIENTS,DATA_ENDPOINT, DATA_ACCESS_KEY, DATA_SECRET_KEY, DATA_BUCKET_NAME= helper_tuple[0], helper_tuple[1], helper_tuple[2], helper_tuple[3], helper_tuple[4], helper_tuple[5], helper_tuple[6], helper_tuple[7], helper_tuple[8], helper_tuple[9], helper_tuple[10], helper_tuple[11]
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



send_params_to_kubernetes_pods(helper())

