# data.py
import os
import torch
import docker

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def _get_data_path():
    """ For test automation using docker-compose. """
    # Figure out FEDn client number from container name
    client = docker.from_env()
    container = client.containers.get(os.environ['HOSTNAME'])
    number = container.name[-1]

    # Return data path
    return f"/var/data/clients/{number}/mnist.pt"

    
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



# Not handled here. Should be done prior to launching the client.

# def get_data(out_dir="data"):
#     """Optional: downloads MNIST if not already there."""
#     if not os.path.exists(f"{out_dir}/train"):
#         torchvision.datasets.MNIST(root=f"{out_dir}/train", train=True, download=True, transform=transforms.ToTensor())
#     if not os.path.exists(f"{out_dir}/test"):
#         torchvision.datasets.MNIST(root=f"{out_dir}/test", train=False, download=True, transform=transforms.ToTensor())

# def split(out_dir="data", n_splits=2):
#     """Optional: create n_splits shards of data."""
#     os.makedirs(f"{out_dir}/clients", exist_ok=True)
#     train_data = torchvision.datasets.MNIST(root=f"{out_dir}/train", train=True, download=True)
#     test_data  = torchvision.datasets.MNIST(root=f"{out_dir}/test", train=False, download=True)

#     def splitset(dataset, parts):
#         n = dataset.shape[0]
#         local_n = floor(n / parts)
#         return [dataset[i * local_n : (i + 1) * local_n] for i in range(parts)]

#     data = {
#         "x_train": splitset(train_data.data, n_splits),
#         "y_train": splitset(train_data.targets, n_splits),
#         "x_test":  splitset(test_data.data, n_splits),
#         "y_test":  splitset(test_data.targets, n_splits),
#     }

#     for i in range(n_splits):
#         subdir = f"{out_dir}/clients/{i+1}"
#         os.makedirs(subdir, exist_ok=True)
#         torch.save({
#             "x_train": data["x_train"][i],
#             "y_train": data["y_train"][i],
#             "x_test":  data["x_test"][i],
#             "y_test":  data["y_test"][i],
#         }, f"{subdir}/mnist.pt")


