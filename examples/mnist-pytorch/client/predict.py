# predict.py
import sys
import torch

from fedn.utils.helpers.helpers import save_metrics
from data import load_data
from model import load_parameters


def predict(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the predict output to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    x_test, y_test = load_data(data_path, is_train=False)
    model = load_parameters(in_model_path)
    model.eval()

    with torch.no_grad():
        logits = model(x_test)
        preds = torch.argmax(logits, dim=1)

    # For demonstration, we just store them as a list:
    result = {"predictions": preds.tolist()}

    save_metrics(result, out_json_path)


if __name__ == "__main__":
    # usage: python predict.py in_model_path out_json_path [data_path]
    in_model = sys.argv[1]
    out_json = sys.argv[2]
    data_path = None
    if len(sys.argv) > 3:
        data_path = sys.argv[3]

    predict(in_model, out_json, data_path)
