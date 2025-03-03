# model.py
import collections
import torch
from fedn.utils.helpers.helpers import get_helper
import numpy
import tempfile
import os
import io
import logging

logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

def compile_model():
    """Compile a PyTorch model for MNIST."""

    torch.manual_seed(42)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(784, 64)
            self.fc2 = torch.nn.Linear(64, 32)
            self.fc3 = torch.nn.Linear(32, 10)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x.reshape(x.size(0), 784)))
            x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x) # switched to CrossEntropyLoss for numerical stability from log_softmax
            return x

    return Net()


def save_parameters(model, out_path):
    """Save model parameters to file.
    
    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)

def load_parameters(model_path):
    """Load model parameters from file and populate a new model.
    
    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """

    # Map numpy arrays back into model.state_dict()
    model = compile_model()
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def save_parameters_to_bytes(model):
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmpf:
        temp_path = tmpf.name
        helper.save(parameters_np, temp_path)
        logger.info(f'temp_path: {temp_path}')
    with open(temp_path, "rb") as f:
        data_bytes = f.read()
    try:
        os.remove(temp_path)
    except OSError:
        pass
    
    logger.info('model saved to bytesio')

    return io.BytesIO(data_bytes)

def load_parameters_from_bytesio(buffer):
    
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmpf:
        temp_path = tmpf.name
        tmpf.write(buffer.getbuffer())
        logger.info(f'temp_path: {temp_path}')
        
    model = compile_model()
    parameters_np = helper.load(temp_path)
    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)

    logger.info('model loaded from bytesio')

    try:
        os.remove(temp_path)
    except OSError:
        pass

    return model

def init_seed(out_path="seed.npz"):
    """ Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    model = compile_model()
    save_parameters(model, out_path)

if __name__ == "__main__":

    init_seed("./seed.npz")
