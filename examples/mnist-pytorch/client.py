import os
import argparse
import uuid
from fedn.network.clients.fedn_client import FednClient, ConnectToApiResult

def on_train(in_model):
    # Here, detect if MALICIOUS=True in env variables:
    malicious = os.environ.get('MALICIOUS', 'False').lower() == 'true'
    attack_type = os.environ.get('ATTACK_TYPE', None)

    # Then do your normal or malicious training
    # out_model, metadata = ...
    return out_model, metadata

def on_validate(in_model):
    # Evaluate model as usual
    metrics = {}
    return metrics

def on_predict(in_model):
    # Possibly unused or do inference
    return {}

def main(api_url, token=None, client_name=None):
    client = FednClient(
        train_callback=on_train,
        validate_callback=on_validate,
        predict_callback=on_predict
    )

    if not client_name:
        client_name = "client-" + str(uuid.uuid4())[:6]

    client.set_name(client_name)
    client_id = str(uuid.uuid4())
    client.set_client_id(client_id)

    controller_config = {
        "name": client_name,
        "client_id": client_id,
        "package": "local",  # or "numpyhelper" or something
        "preferred_combiner": "",  # if you have multiple combiners
    }

    # Connect to aggregator
    result, combiner_config = client.connect_to_api(api_url, token, controller_config)

    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to aggregator!")
        return

    # Initialize gRPC
    success = client.init_grpchandler(combiner_config, client_id, token=token)
    if not success:
        print("Failed to init gRPC!")
        return

    # Start the client loop
    client.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, required=True)
    parser.add_argument("--token", type=str, required=False)
    parser.add_argument("--client-name", type=str, required=False)
    args = parser.parse_args()

    main(args.api_url, args.token, args.client_name)
