import numpy as np
from sklearn.linear_model import SGDClassifier
from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics
import tempfile
import os
import io
import collections
import logging

logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)

HELPER_MODULE = 'numpyhelper'
helper = get_helper(HELPER_MODULE)

def compile_model():
    """ Compile the sklearn model.

    :return: The compiled model.
    :rtype: sklearn.linear_model._logistic.LogisticRegression
    """

    model = SGDClassifier(warm_start=True, loss='log_loss', max_iter=20, learning_rate='invscaling', eta0=0.001, random_state=100)
    model.fit([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 1, 2])

    return model

def save_parameters(model, out_path):
    """ Save model paramters to file.

    :param model: The model to serialize.
    :type model: sklearn.linear_model._logistic.LogisticRegression
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)

    helper.save(parameters_np, out_path)

def save_parameters_to_bytes(model):
    parameters_np = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmpf:
        temp_path = tmpf.name
        helper.save(parameters_np, temp_path)

    with open(temp_path, "rb") as f:
        data_bytes = f.read()
    try:
        os.remove(temp_path)
    except OSError:
        pass
    
    logger.info('model saved to bytesio')

    return io.BytesIO(data_bytes)


def load_parameters(model_path):
    """ Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = compile_model()
    parameters_np = np.array(helper.load(model_path))

    model.coef_ = parameters_np[:, 0:4]
    model.intercept_ = parameters_np[:, -1]

    return model

def load_parameters_from_bytesio(buffer):
    
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmpf:
        temp_path = tmpf.name
        tmpf.write(buffer.getbuffer())

    model = compile_model()
    parameters_np = np.array(helper.load(temp_path))

    model.coef_ = parameters_np[:, 0:4]
    model.intercept_ = parameters_np[:, -1]
    logger.info('model loaded from bytesio')

    try:
        os.remove(temp_path)
    except OSError:
        pass

    return model


def init_seed(out_path='seed.npz'):
    """ Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()

    save_parameters(model, out_path)

if __name__ == "__main__":

    init_seed("./seed.npz")
