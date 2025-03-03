#!./.mnist-pytorch/bin/python
import os
from math import floor

import fire
import torch
import torchvision


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n/parts)
    result = []
    for i in range(parts):
        result.append(dataset[i*local_n: (i+1)*local_n])
    return result


def split(n_splits=2):
    out_dir = './data'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    clients_dir = f'{out_dir}/clients'
    if not os.path.exists(clients_dir):
        os.mkdir(clients_dir)

    train_data = torchvision.datasets.MNIST(
        root=f'{out_dir}/train', transform=torchvision.transforms.ToTensor, train=True, download=True)
    test_data = torchvision.datasets.MNIST(
        root=f'{out_dir}/test', transform=torchvision.transforms.ToTensor, train=False, download=True)
    data = {
        'x_train': splitset(train_data.data, n_splits),
        'y_train': splitset(train_data.targets, n_splits),
        'x_test': splitset(test_data.data, n_splits),
        'y_test': splitset(test_data.targets, n_splits),
    }

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



# if __name__ == '__main__':
#     fire.Fire(split)
