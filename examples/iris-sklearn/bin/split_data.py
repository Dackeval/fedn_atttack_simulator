#!./.iris-sklearn/bin/python
import os
import json
from math import floor
import fire
from minio import Minio
from minio.error import S3Error

def splitset(dataset, parts):
    n = len(dataset)
    print('length of dataset: ', n)
    local_n = floor(n/parts)
    result = []
    for i in range(parts):
        result.append(dataset[i*local_n: (i+1)*local_n])
    return result


def split(n_splits, data_endpoint, data_access_key, data_secret_key, data_bucket_name):
    out_dir = './data'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    clients_dir = f'{out_dir}/clients'
    if not os.path.exists(clients_dir):
        os.mkdir(clients_dir)

    with open("iris_data/iris.json", 'r') as json_file:
        data = json.load(json_file)

    data_splits = {
        'x_train': splitset(data['x_train'], n_splits),
        'y_train': splitset(data['y_train'], n_splits),
        'x_test': splitset(data['x_test'], n_splits),
        'y_test': splitset(data['y_test'], n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = f'{out_dir}/clients/{str(i+1)}'
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        iris_data_dict = {
            'x_train': data_splits['x_train'][i],
            'y_train': data_splits['y_train'][i],
            'x_test': data_splits['x_test'][i],
            'y_test': data_splits['y_test'][i]
        }
            
        with open(f'{subdir}/iris.json', "w") as json_file:
            json.dump(iris_data_dict, json_file)
        print('Split data saved to:', subdir)

    push_to_bucket(clients_dir, data_endpoint, data_access_key, data_secret_key, data_bucket_name)

def push_to_bucket(data_path, data_endpoint, data_access_key, data_secret_key, data_bucket_name):
    """
    Push data partitions to bucket to make available for train + validation
    """
    # _, port = data_endpoint.split(":")
    # data_endpoint = "localhost:" + port
    minio_client = Minio(str(data_endpoint),
        access_key=str(data_access_key),
        secret_key=str(data_secret_key),
        secure=False)
    
    # user inputted bucket name

    # check for existing bucket or create
    if not minio_client.bucket_exists(data_bucket_name):
        minio_client.make_bucket(data_bucket_name)
        print(f"Bucket '{data_bucket_name}' created")
    else:
        print(f"Bucket '{data_bucket_name}' already created")
    
    for client_id in os.listdir(data_path):
        partition_path = os.path.join(data_path, str(client_id), "iris.json")

        # check for file before upload
        if os.path.isfile(partition_path):
            obj_name = f"clients/{client_id}/iris.json"

            minio_client.fput_object(bucket_name=data_bucket_name, 
                                     object_name=obj_name, 
                                     file_path=partition_path)

            print(f"Uploaded '{partition_path}' to '{data_bucket_name}'/'{obj_name}'")

    print('All partitions was uploaded successfully!')
