# import torch
# import math

# def compile_model():
#     """Compile a PyTorch model for MNIST."""

#     torch.manual_seed(42)

#     class Net(torch.nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.fc1 = torch.nn.Linear(784, 64)
#             self.fc2 = torch.nn.Linear(64, 32)
#             self.fc3 = torch.nn.Linear(32, 10)

#         def forward(self, x):
#             x = torch.nn.functional.relu(self.fc1(x.reshape(x.size(0), 784)))
#             x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
#             x = torch.nn.functional.relu(self.fc2(x))
#             x = self.fc3(x) # switched to CrossEntropyLoss for numerical stability from log_softmax
#             return x

#     return Net()

# def load_data(is_train=True):
#     """ Load data from disk.

#     :param data_path: Path to data file.
#     :type data_path: str
#     :param is_train: Whether to load training or test data.
#     :type is_train: bool
#     :return: Tuple of data and labels.
#     :rtype: tuple
#     """
#     data = torch.load("/Users/sigvard/Desktop/fedn_attack_simulator/examples/mnist-pytorch/data/mnist/noniid_unbalanced/clients/1/mnist.pt")

#     if is_train:
#         X = data['x_train']
#         y = data['y_train']
#     else:
#         X = data['x_test']
#         y = data['y_test']

#     X = torch.tensor(X)        
#     y = torch.tensor(y)

#     # Normalize
#     X = X / 255

#     return X, y

# def train():
#     model = compile_model()
#     x_train, y_train = load_data(is_train=True)
#     # Train
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#     n_batches = int(math.ceil(len(x_train) / 32))
#     criterion = torch.nn.CrossEntropyLoss()
#     for e in range(5):  # epoch loop
#         for b in range(n_batches):  # batch loop
#             # Retrieve current batch
#             batch_x = x_train[b * 32:(b + 1) * 32]
#             batch_y = y_train[b * 32:(b + 1) * 32]

#             # Train on batch
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
    

# if __name__ == "__main__":
#     train()

import os
import torch
import matplotlib.pyplot as plt

def check_partitions(partitions_dir):
    """
    For each client partition under `partitions_dir`,
    - Load mnist.pt
    - Print shapes of train/test sets
    """
    clients = sorted(os.listdir(partitions_dir))

    for client_id in clients:
        partition_path = os.path.join(partitions_dir, client_id, "mnist.pt")
        if not os.path.isfile(partition_path):
            continue  # skip if no 'mnist.pt'

        data = torch.load(partition_path)

        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        # Convert to tensors if they're numpy arrays
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train)
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test)

        print(f"\n=== Client {client_id} ===")
        print(f"x_train shape: {tuple(x_train.shape)}  y_train shape: {tuple(y_train.shape)}")
        print(f"x_test  shape: {tuple(x_test.shape)}  y_test  shape: {tuple(y_test.shape)}")

        # Show label distribution in train split
        label_counts = torch.bincount(y_train) if y_train.dim() == 1 else None
        if label_counts is not None:
            print("Train label distribution:")
            for label_idx, count in enumerate(label_counts):
                print(f"  Label {label_idx}: {count.item()} samples")
        else:
            print("Label distribution not shown (y_train not 1D).")


        if x_train.numel() > 0:
            sample_img = x_train[0]
            sample_label = y_train[0].item() if y_train.numel() > 0 else None

            # If shape is (28, 28), good. If shape is (1, 28, 28), squeeze it
            if sample_img.dim() == 3 and sample_img.shape[0] == 1:
                sample_img = sample_img.squeeze(0)

            sample_img = sample_img.float() / 255.0  # normalize to [0,1] if not already
            plt.imshow(sample_img, cmap="gray")
            plt.title(f"Client {client_id} - Example label: {sample_label}")
            plt.show()


if __name__ == "__main__":
    PARTITIONS_DIR = "/Users/sigvard/Desktop/fedn_attack_simulator/examples/mnist-pytorch/data/mnist/noniid_unbalanced/clients"  # <-- adjust to your path
    check_partitions(PARTITIONS_DIR)
