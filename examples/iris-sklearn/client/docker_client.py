import argparse
import uuid
from fedn.network.clients.fedn_client import FednClient, ConnectToApiResult
from train import train
from validate import validate
from model import load_parameters_from_bytesio, save_parameters_to_bytes
import random
import string
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fedn")

def get_api_url(api_url: str, api_port: int):
    url = f"{api_url}:{api_port}" if api_port else api_url
    if not url.endswith("/"):
        url += "/"
    return url

def generate_variable_name(length=8):
    first_char = random.choice(string.ascii_letters + "_")  # First character: letter or underscore
    other_chars = ''.join(random.choices(string.ascii_letters + string.digits + "_", k=length-1))  # Remaining characters
    return first_char + other_chars


def on_train(in_model_bytes, client_settings=None):
    logger.info('in_model_bytes: %s', in_model_bytes)
    logger.info('in_model type: %s', type(in_model_bytes))
    
    in_model = load_parameters_from_bytesio(in_model_bytes)

    logger.info('Loaded in_model: %s', in_model)
    logger.info('Loaded in_model type: %s', type(in_model))
    metadata, out_model = train(in_model)
    metadata = {"training_metadata": metadata}
    out_model_bytesIO = save_parameters_to_bytes(out_model)
    logger.info('train sending out_model_bytesIO')
    return out_model_bytesIO, metadata

def on_validate(in_model_bytes):

    logger.info('Validating model')
    in_model = load_parameters_from_bytesio(in_model_bytes)
    logger.info('Validatin complete')
    metrics = validate(in_model)

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

    name = generate_variable_name()
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
    parser.add_argument("--api-url", type=str, required=True, help="The API URL")
    parser.add_argument("--api-port", type=int, required=False, help="The API Port")
    parser.add_argument("--token", type=str, required=False, help="The API Token")

    args = parser.parse_args()
    main(args.api_url, args.api_port, args.token)
