# data.py
import os
import torch


dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def _get_data_path():   
    # # # for local docker testing
    # return "/app/client/data/clients/1/mnist.pt"
    client_index = os.environ.get("CLIENT_INDEX", "1")
    return f"/app/data/clients/{client_index}/mnist.pt"

    
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
        data = torch.load(_get_data_path())
    else:
        data = torch.load(data_path)

    if is_train:
        X = data['x_train']
        y = data['y_train']
    else:
        X = data['x_test']
        y = data['y_test']

    # Normalize
    X = X / 255

    return X, y

if __name__ == "__main__":
    load_data(_get_data_path())
