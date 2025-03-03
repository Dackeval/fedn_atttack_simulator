# validate.py
import os
import sys
import torch
import time

from fedn.utils.helpers.helpers import save_metrics
from data import load_data
from model import load_parameters, save_parameters 
import logging

logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

def validate(in_model, out_json_path='/app/validation.json', data_path=None):
    """Validate model with training and test partitions.
    
    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """

    # # double check this line.
    # client_name = os.uname().nodename

    x_train, y_train = load_data(data_path, is_train=True)
    x_test, y_test = load_data(data_path, is_train=False)

    #model = load_parameters(in_model_path)
    


    #print(client_name == "benign_client1")
    
    # # ----------------------- Maybe not needed -----------------------
    # if client_name == "benign_client1":
    #     logger.info(f"Client is {client_name}")
    #     current_time = int(time.time())
            
    #     # Ensure the directory exists
    #     os.makedirs("/var/parameter_store/models", exist_ok=True)
        
    #     # Save the model parameters
    #     save_parameters(model, out_path=f"/var/parameter_store/models/{current_time}.npz")

    # -----------------------------------------------------------------
    client_index = os.environ.get("CLIENT_INDEX", "1")
    model = load_parameters(f"/app/model_update_{client_index}.npz")
    logger.info(f"/app/model_update_{client_index}.npz")
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        # Training set
        train_out = model(x_train)
        train_loss = criterion(train_out, y_train)
        train_acc = torch.sum(torch.argmax(train_out, dim=1) == y_train).item() / len(train_out)

        # Test set
        test_out = model(x_test)
        test_loss = criterion(test_out, y_test)
        test_acc = torch.sum(torch.argmax(test_out, dim=1) == y_test).item() / len(test_out)

    metrics = {
        "training_loss": train_loss.item(),
        "training_accuracy": train_acc,
        "test_loss": test_loss.item(),
        "test_accuracy": test_acc,
    }

    logger.info('Training accuracy: %.4f', train_acc)
    logger.info('Training loss: %.4f', train_loss.item())
    logger.info('Test accuracy: %.4f', test_acc)
    logger.info('Test loss: %.4f', test_loss.item())   

    save_metrics(metrics, out_json_path)
    logger.info('Validation done.')
    return metrics

if __name__ == "__main__":
    # usage: python validate.py in_model_path out_json_path [data_path]
    in_model = sys.argv[1]
    out_json = sys.argv[2]
    data_path = None
    if len(sys.argv) > 3:
        data_path = sys.argv[3]
    validate(in_model, out_json, data_path)
