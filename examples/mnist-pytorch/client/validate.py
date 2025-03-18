# validate.py
import os
import sys
import torch
from fedn.utils.helpers.helpers import save_metrics
from data import load_data
from model import load_parameters 
import logging

# for debugging
logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)

# for executing in correct directory
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

def validate(global_model, out_json_path='/app/validation.json', data_path=None):
    """Validate model with training and test partitions.
    
    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """

    x_train, y_train = load_data(is_train=True)
    x_test, y_test = load_data(is_train=False)


    client_index = os.environ.get("CLIENT_ID", "1")
    local_model = load_parameters(f"/app/model_update_{client_index}.npz")
    
    local_model.eval()
    global_model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        # Training set
        train_out = local_model(x_train)
        train_loss = criterion(train_out, y_train)
        train_acc = torch.sum(torch.argmax(train_out, dim=1) == y_train).item() / len(train_out)

        # Test set
        test_out = global_model(x_test)
        test_loss = criterion(test_out, y_test)
        test_acc = torch.sum(torch.argmax(test_out, dim=1) == y_test).item() / len(test_out)

    metrics = {
        "training_loss": train_loss.item(),
        "training_accuracy": train_acc,
        "test_loss": test_loss.item(),
        "test_accuracy": test_acc,
    }

    # for debugging
    logger.info('Training accuracy: %.4f', train_acc)
    logger.info('Training loss: %.4f', train_loss.item())
    logger.info('Test accuracy: %.4f', test_acc)
    logger.info('Test loss: %.4f', test_loss.item())   

    save_metrics(metrics, out_json_path)
    logger.info('Validation done.')
    return metrics