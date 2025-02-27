# parameter_store.py
import os
import json


def create_parameter_store(
    ben_count, mal_count, attack_type, defence_type, host, token,
    lr, epochs, batch_size, inflation_factor
):
    dir_name = "parameter_store"
    file_name = "param_store.json"

    # Clean up any old store if it exists
    if os.path.exists(dir_name):
        for root, dirs, files in os.walk(dir_name, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(dir_name)
        print(f"Directory '{dir_name}' and its contents have been deleted.")

    os.makedirs(dir_name)
    print(f"Directory '{dir_name}' created.")

    os.makedirs(f"{dir_name}/models")
    print(f"Directory '{dir_name}/models' created.")

    clients = []

    # Add benign clients
    for i in range(ben_count):
        client_id = i + 1
        client_entry = {
            "client_id": client_id,
            "is_malicious": False
        }
        clients.append(client_entry)

    # Add malicious clients
    for i in range(mal_count):
        client_id = ben_count + i + 1
        client_entry = {
            "client_id": client_id,
            "is_malicious": True,
            "attack_type": attack_type,
            "inflation_factor": inflation_factor
        }
        clients.append(client_entry)

    store = {
        "ben_count": ben_count,
        "mal_count": mal_count,
        "attack_type": attack_type,
        "defence_type": defence_type,
        "host": host,
        "token": token,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "inflation_factor": inflation_factor,

        "clients": clients
    }

    path = f"{dir_name}/{file_name}"
    if os.path.isfile(path):
        os.remove(path)
        print(f"File {file_name} deleted.")

    with open(path, 'w') as file:
        json.dump(store, file, indent=4)
        print(f"File {file_name} created and data written.")


# if __name__ == '__main__':
#     fire.Fire({
#         'create_parameter_store': create_parameter_store,
#         # 'delete_parameter_store': delete_parameter_store,
#     })