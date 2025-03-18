# data.py
import os
import torch
from minio import Minio
from minio.error import S3Error
import logging

# for debugging
logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def _get_data_path():   

    client_index = os.environ.get("CLIENT_ID", "1")
    remote_data_path = f"clients/{client_index}/mnist.pt"
    client_data_directory_path = "/app/data/clients"
    partition_path = f"{client_data_directory_path}/{client_index}/mnist.pt"
    if os.path.isdir(f"{client_data_directory_path}/{client_index}"):
        return remote_data_path, partition_path, True
    else:
        os.makedirs(f"{client_data_directory_path}/{client_index}", exist_ok=True)
        return remote_data_path, partition_path, False

def _fetch_data_partition():
    # fetch data from data_endpoint
    data_bucket_name =  str(os.environ.get("DATA_BUCKET_NAME", ""))
    remote_data_path, partition_path, exists = _get_data_path()
    if exists:
       logger.info(f"Partition already exists: {partition_path}")
    else: 
        minio_client = Minio(
        str(os.environ.get("DATA_ENDPOINT", "")),
        access_key=str(os.environ.get("DATA_ACCESS_KEY", "")),
        secret_key=str(os.environ.get("DATA_SECRET_KEY", "")),
        secure=False
    )   
        try:
            minio_client.fget_object(
                bucket_name=data_bucket_name,
                object_name=remote_data_path,
                file_path=partition_path

            )
        except S3Error as err:
            logger.warning(f"Failed to download partition: {err}")
            raise 

    return partition_path
    
def load_data(is_train=True):
    """ Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    data = torch.load(_fetch_data_partition())

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
    load_data()
