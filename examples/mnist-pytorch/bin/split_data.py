#!./.mnist-pytorch/bin/python
import os
from math import floor
import fire
import torch
import torchvision
from minio import Minio
from minio.error import S3Error
import numpy as np
from collections import defaultdict

def splitset(x, y, num_clients, balanced, seed=42):
    num_classes = len(np.unique(y))
    total_samples = len(x)
    np.random.seed(seed)
    x_partitioned = []
    y_partitioned = []

    local_n = floor(total_samples/num_clients)
    for i in range(num_clients):
        x_partitioned.append(x[i*local_n: (i+1)*local_n])
        y_partitioned.append(y[i*local_n: (i+1)*local_n])

    return x_partitioned, y_partitioned


def iid_unbalanced_split(x, y, num_clients, min_samples_prop=0.1, seed=42):
    """
    Returns x_list, y_list for 'IID unbalanced':
    - The label distribution is the same for all clients (like full dataset),
    - The total # of samples per client can differ (unbalanced).
    """
    np.random.seed(seed)

    total_samples = len(x)
    num_classes = len(np.unique(y))

    # STEP 1: Decide how many total samples each client gets
    # (We sample from Dirichlet, but ensure no client is under min_samples)
    client_sizes = np.random.dirichlet(alpha=np.ones(num_clients)) * total_samples
    client_sizes = client_sizes.astype(int)
    min_samples = total_samples / num_clients * min_samples_prop

    while np.any(client_sizes < min_samples):
        client_sizes = np.random.dirichlet(alpha=np.ones(num_clients)) * total_samples
        client_sizes = client_sizes.astype(int)

    # STEP 2: Count how many of each class there is in the entire dataset
    class_indices = {}
    for c in range(num_classes):
        class_indices[c] = np.where(y == c)[0]
        np.random.shuffle(class_indices[c])
    class_counts = {c: len(class_indices[c]) for c in range(num_classes)}

    # Keep track of where we are for each class as we assign to each client
    class_ptrs = {c: 0 for c in range(num_classes)}

    x_list = []
    y_list = []

    # For each client i, pick the same fraction of each class
    for i in range(num_clients):
        n_i = client_sizes[i]  # total samples for client i

        # fraction of each class for the entire dataset
        # multiply that fraction by n_i
        x_client = []
        y_client = []

        for c in range(num_classes):
            # ideal count for class c is
            ideal_count = int(round((class_counts[c] / total_samples) * n_i))

            # get the next 'ideal_count' samples from class c's indices
            start = class_ptrs[c]
            end = start + ideal_count

            # guard if we run out of indices
            end = min(end, len(class_indices[c]))

            selected_indices = class_indices[c][start:end]
            class_ptrs[c] = end  # move the pointer

            # accumulate
            x_client.append(x[selected_indices])
            y_client.append(y[selected_indices])

        x_list.append(torch.from_numpy(np.concatenate(x_client, axis=0)))
        y_list.append(torch.from_numpy(np.concatenate(y_client, axis=0)))

    return x_list, y_list


def dirichlet_label_skew_split(x, y, balanced, num_clients, alpha=0.5, seed=42):
    """
    Non-IID + Unbalanced: Dirichlet over label distribution AND per-client sample count.
    """
    np.random.seed(seed)

    num_classes = len(np.unique(y))
    total_samples = len(x)
    if balanced:
        samples_per_client = len(x) // num_clients
        client_sizes = np.array([samples_per_client] * num_clients)    
    else:
        min_samples_prop = 0.1
        min_samples = (total_samples / num_clients) * min_samples_prop # minimum samples per client

        # how many samples each client gets 
        client_sizes = np.random.dirichlet(alpha=np.ones(num_clients)) * total_samples
        client_sizes = client_sizes.astype(int)

        # minimum number of samples per client
        while np.any(client_sizes < min_samples):
            client_sizes = np.random.dirichlet(alpha=np.ones(num_clients)) * total_samples
            client_sizes = client_sizes.astype(int)
    

    class_indices = {c: np.where(y == c)[0] for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])
    class_ptr = {c: 0 for c in range(num_classes)}

    x_list = []
    y_list = []

    for i in range(num_clients):
        n_samples = client_sizes[i]
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
        class_counts = (proportions * n_samples).astype(int)

        # fix rounding errors
        while class_counts.sum() < n_samples:
            class_counts[np.random.randint(0, num_classes)] += 1
        while class_counts.sum() > n_samples:
            class_counts[np.random.randint(0, num_classes)] -= 1

        x_client = []
        y_client = []

        for c in range(num_classes):
            count = class_counts[c]
            start = class_ptr[c]
            end = start + count
            selected_indices = class_indices[c][start:end]

            x_client.append(x[selected_indices])
            y_client.append(y[selected_indices])

            class_ptr[c] += count

        x_list.append(np.concatenate(x_client, axis=0))
        y_list.append(np.concatenate(y_client, axis=0))

    return x_list, y_list



def split(n_splits, data_endpoint, data_access_key, data_secret_key, data_bucket_name, iid, balanced):

    # params to be set, possibly update to prompt user for these
    seed = 42
    out_dir = './data/mnist'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    train_data = torchvision.datasets.MNIST(
        root=f'{out_dir}/train', transform=torchvision.transforms.ToTensor, train=True, download=True)
    test_data = torchvision.datasets.MNIST(
        root=f'{out_dir}/test', transform=torchvision.transforms.ToTensor, train=False, download=True)
    
    # for IID and balanced
    if iid == "iid" and balanced == "balanced":
        balanced_bool = True
        out_dir = f'{out_dir}/iid_balanced'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        x_train, y_train = splitset(train_data.data, train_data.targets, n_splits, balanced_bool)
        x_test, y_test = splitset(test_data.data, test_data.targets, n_splits, balanced_bool)
        data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
        }
        clients_dir = f'{out_dir}/clients'
        if not os.path.exists(clients_dir):
            os.mkdir(clients_dir)
        for i in range(n_splits):
            subdir = f'{out_dir}/clients/{str(i+1)}'
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            torch.save({
                'x_train': data['x_train'][i],
                'y_train': data['y_train'][i],
                'x_test': data['x_test'][i],
                'y_test': data['y_test'][i],
            },
                f'{subdir}/mnist.pt')
            print('Split data saved to:', subdir)
    # for non-IID and balanced
    if iid == "noniid" and balanced == "balanced":
        balanced_bool = True
        out_dir = f'{out_dir}/noniid_balanced'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # split data into n_splits
        x_train, y_train = dirichlet_label_skew_split(train_data.data, train_data.targets, balanced_bool, n_splits, alpha=0.5, seed=42)
        x_test, y_test = dirichlet_label_skew_split(test_data.data, test_data.targets, balanced_bool, n_splits, alpha=0.5, seed=42) 
        data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
        }
        clients_dir = f'{out_dir}/clients'
        if not os.path.exists(clients_dir):
            os.mkdir(clients_dir)
        for i in range(n_splits):
            subdir = f'{out_dir}/clients/{str(i+1)}'
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            torch.save({
                'x_train': data['x_train'][i],
                'y_train': data['y_train'][i],
                'x_test': data['x_test'][i],
                'y_test': data['y_test'][i],
            },
                f'{subdir}/mnist.pt')
            print('Split data saved to:', subdir)
    # for IID and non-balanced
    if iid == "iid" and balanced == "unbalanced":
        balanced_bool = False
        out_dir = f'{out_dir}/iid_unbalanced'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        x_train, y_train = iid_unbalanced_split(train_data.data, train_data.targets, n_splits)
        x_test, y_test = iid_unbalanced_split(test_data.data, test_data.targets, n_splits)
        data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
        }
        clients_dir = f'{out_dir}/clients'
        if not os.path.exists(clients_dir):
            os.mkdir(clients_dir)

        for i in range(n_splits):
            subdir = f'{out_dir}/clients/{str(i+1)}'
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            torch.save({
                'x_train': data['x_train'][i],
                'y_train': data['y_train'][i],
                'x_test': data['x_test'][i],
                'y_test': data['y_test'][i],
            },
                f'{subdir}/mnist.pt')
            print('Split data saved to:', subdir)
            
    # for non-IID and non-balanced
    if iid == "noniid" and balanced == "unbalanced":
        balanced_bool = False

        out_dir = f'{out_dir}/noniid_unbalanced'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # split data into n_splits
        x_train, y_train = dirichlet_label_skew_split(train_data.data, train_data.targets, balanced_bool, n_splits, alpha=0.5, seed=42)
        x_test, y_test = dirichlet_label_skew_split(test_data.data, test_data.targets, balanced_bool, n_splits, alpha=0.5, seed=42) 
        data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
        }

        clients_dir = f'{out_dir}/clients'
        if not os.path.exists(clients_dir):
            os.mkdir(clients_dir)

        for i in range(n_splits):
            subdir = f'{out_dir}/clients/{str(i+1)}'
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            torch.save({
                'x_train': data['x_train'][i],
                'y_train': data['y_train'][i],
                'x_test': data['x_test'][i],
                'y_test': data['y_test'][i],
            },
                f'{subdir}/mnist.pt')
            print('Split data saved to:', subdir)

    print(f"Data split into {n_splits} clients with {iid} and {balanced} data.")

    push_to_bucket(clients_dir, data_endpoint, data_access_key, data_secret_key, data_bucket_name, iid, balanced)

def push_to_bucket(data_path, data_endpoint, data_access_key, data_secret_key, data_bucket_name, iid_str, balanced_str):
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
        partition_path = os.path.join(data_path, str(client_id), "mnist.pt")

        # check for file before upload
        if os.path.isfile(partition_path):
            obj_name = f"mnist/{iid_str}_{balanced_str}/clients/{client_id}/mnist.pt"

            minio_client.fput_object(bucket_name=data_bucket_name, 
                                     object_name=obj_name, 
                                     file_path=partition_path)

            print(f"Uploaded '{partition_path}' to '{data_bucket_name}'/'{obj_name}'")

    print('All partitions was uploaded successfully!')

if __name__ == "__main__":
    split(10, "localhost:9000", "minioadmin", "minioadmin", "fedn")