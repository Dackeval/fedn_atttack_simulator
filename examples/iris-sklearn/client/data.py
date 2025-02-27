# data.py
import os
import docker
import json


dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def _get_data_path():
    """ For test automation using docker-compose. """
    # Figure out FEDn client number from container name
    # client = docker.from_env()
    # container = client.containers.get(os.environ['HOSTNAME'])
    # number = container.name[-1]
    number = 1 
    # Return data path
    return "/app/client/data/clients/1/iris.json"

def load_data(data_path, is_train=True):
    """ Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """

    if data_path is None:
        with open(_get_data_path(), 'r') as json_file:
            data = json.load(json_file)
    else:
        with open(data_path, 'r') as json_file:
            data = json.load(json_file)

    if is_train:
        X = data['x_train']
        y = data['y_train']
    else:
        X = data['x_test']
        y = data['y_test']

    # Normalize - Do we normalize?

    return X, y

if __name__ == "__main__":
    load_data(_get_data_path())
