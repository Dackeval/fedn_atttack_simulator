from fedn import APIClient
import time
import docker
import sys
import subprocess
import json
sys.path.append('/home/ubuntu/fedn-attack-sim-uu/examples/mnist-pytorch')

def start_clients(combiner_ip, benign_client_count, malicious_client_count, attack_type):
    script_path = './bin/start_clients.sh'  # Path to your shell script
    client = docker.from_env()

    try:
        # Run the shell script with the provided arguments
        print("running the shell script")
        result = subprocess.run(f"{script_path} {combiner_ip} {int(benign_client_count)} {int(malicious_client_count)} {attack_type}", shell=True, check=True, text=True, capture_output=True)
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

COMBINER_IP = get_combiner_ip()

print(f"Combiner IP: {COMBINER_IP} is set")

DISCOVER_HOST = COMBINER_IP
DISCOVER_PORT = 8092
client = APIClient(DISCOVER_HOST, DISCOVER_PORT)

print(f"API Client connected to combiner at: {DISCOVER_HOST}:{DISCOVER_PORT}")

# CLIENTS
docker_client = docker.from_env()
running_containers = docker_client.containers.list()

if len(running_containers) != 0:
    print(f"{len(running_containers)} clients are running!")
    for id, container in enumerate(running_containers):
        print(f"{id} - {container.name}")
else:
    print("No containers are running!")
    benign_count = int(input("Number of benign clients: "))
    malicious_count = int(input("Number of malicious clients: "))
    attack_type = input("Define attack type: ")
    start_clients(COMBINER_IP, benign_count, malicious_count, attack_type)

sessions_list = client.list_sessions()['result']
if len(sessions_list) != 0:
    print(f"Number of sessions: {len(sessions_list)}")
    for id, session in enumerate(sessions_list):
        print(f"{id} - {session['session_id']}")
else:
    print("No sessions yet")

client._get_url('set_package')

client.set_package('package.tgz', 'numpyhelper')
client.set_initial_model('seed.npz')
seed_model = client.get_initial_model()

session_id = input("Set session_id: ")
rounds = input("Set the number of rounds: ")

session_config_fedavg = {
    "helper": "numpyhelper",
    "session_id": session_id,
    "aggregator": "fedavg",
    "model_id": seed_model['model_id'],
    "rounds": int(rounds)
}

result_fedavg = client.start_session(**session_config_fedavg)
time.sleep(10)

def run_until_finished(session_id):
    while not client.session_is_finished(session_id):
        models = client.list_models(session_id)
        print(f"Rounds: {models['count']} out of {session_config_fedavg['rounds']} completed!", end="\r")
        time.sleep(15)

# Call the function
run_until_finished(session_id) 

if client.session_is_finished(session_id):
    print(f"The session: {session_id} is over!")
    kill_clients()