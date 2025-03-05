#!./.iris-sklearn/bin/python
import os
import json
from math import floor
import fire

def splitset(dataset, parts):
    n = len(dataset)
    print('length of dataset: ', n)
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

# if __name__ == '__main__':
#     fire.Fire(split)
