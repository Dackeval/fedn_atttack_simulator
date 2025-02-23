import os
import argparse
import uuid
from fedn.network.clients.fedn_client import FednClient, ConnectToApiResult

# def on_train(in_model):
#     # Here, detect if MALICIOUS=True in env variables:
#     malicious = os.environ.get('MALICIOUS', 'False').lower() == 'true'
#     attack_type = os.environ.get('ATTACK_TYPE', None)

#     # Then do your normal or malicious training
#     # out_model, metadata = ...
#     return out_model, metadata

# def on_validate(in_model):
#     # Evaluate model as usual
#     metrics = {}
#     return metrics

# def on_predict(in_model):
#     # Possibly unused or do inference
#     return {}

# def main(api_url, token=None, client_name=None):
#     client = FednClient(
#         train_callback=on_train,
#         validate_callback=on_validate,
#         predict_callback=on_predict
#     )

#     if not client_name:
#         client_name = "client-" + str(uuid.uuid4())[:6]

#     client.set_name(client_name)
#     client_id = str(uuid.uuid4())
#     client.set_client_id(client_id)

#     controller_config = {
#         "name": client_name,
#         "client_id": client_id,
#         "package": "local",  # or "numpyhelper" or something
#         "preferred_combiner": "",  # if you have multiple combiners
#     }

#     # Connect to aggregator
#     result, combiner_config = client.connect_to_api(api_url, token, controller_config)

#     if result != ConnectToApiResult.Assigned:
#         print("Failed to connect to aggregator!")
#         return

#     # Initialize gRPC
#     success = client.init_grpchandler(combiner_config, client_id, token=token)
#     if not success:
#         print("Failed to init gRPC!")
#         return

#     # Start the client loop
#     client.run()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--api-url", type=str, required=True)
#     parser.add_argument("--token", type=str, required=False)
#     parser.add_argument("--client-name", type=str, required=False)
#     args = parser.parse_args()

#     main(args.api_url, args.token, args.client_name)


import argparse

from fedn.network.clients.fedn_client import FednClient, ConnectToApiResult


def get_api_url(api_url: str, api_port: int):
    url = f"{api_url}:{api_port}" if api_port else api_url
    if not url.endswith("/"):
        url += "/"
    return url

def on_train(in_model):
    training_metadata = {
        "num_examples": 1,
        "batch_size": 1,
        "epochs": 1,
        "lr": 1,
    }

    metadata = {"training_metadata": training_metadata}

    # Do your training here, out_model is your result...
    out_model = in_model

    return out_model, metadata

def on_validate(in_model):
    # Calculate metrics here...
    metrics = {
        "test_accuracy": 0.9,
        "test_loss": 0.1,
        "train_accuracy": 0.8,
        "train_loss": 0.2,
    }
    return metrics

def on_predict(in_model):
    # Do your prediction here...
    prediction = {
        "prediction": 1,
        "confidence": 0.9,
    }
    return prediction


def main(api_url: str, api_port: int, token: str = None):
    fedn_client = FednClient(train_callback=on_train, validate_callback=on_validate, predict_callback=on_predict)

    url = get_api_url(api_url, api_port)

    name = input("Enter Client Name: ")
    fedn_client.set_name(name)

    client_id = str(uuid.uuid4())
    fedn_client.set_client_id(client_id)

    controller_config = {
        "name": name,
        "client_id": client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, combiner_config = fedn_client.connect_to_api(url, token, controller_config)

    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        return

    result: bool = fedn_client.init_grpchandler(config=combiner_config, client_name=client_id, token=token)

    if not result:
        return

    fedn_client.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client Example")
    parser.add_argument("api.fedn.scaleoutsystems.com/simulator-test-1-usn-fedn-reducer", True, type=str, help="The API URL")
    parser.add_argument("--api-port", type=int, required=False, help="The API Port")
    parser.add_argument("--token", type=str, required=False, help="The API Token")

    args = parser.parse_args()
    main(args.api_url, args.api_port, args.token)
