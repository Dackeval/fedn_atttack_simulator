import torch
import logging

logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)

def label_flip(y):
    """
    Label flipping attack - basic
    """
    unique_labels = y.unique().tolist()
    for old_lbl in unique_labels:
        new_lbl = (old_lbl + 1) % 10
        y[y == old_lbl] = new_lbl
    return y

def backdoor_35int(x_train, y_train):
    """
    Backdoor attack with intensity 0.35
    """
    target_label = 8 # Label which we try to misclassify as the backdoor label
    logger.info(f"[Attack Training]: Running a backdoor attack to misclassify label {target_label}")

    # Inject backdoor to backdoor label
    for index, is_target in enumerate((y_train == target_label).tolist()):
        if is_target:
            # modify the second row, ie. index 1, to have an pixel intensity close to 1, ie white, across all 28 pixels.
            x_train[index][1] = torch.tensor([0.9922 for x in range(28)]) 

    prop_counter = 0 
    bd_prop = 0.2 # backdoor probability for other labels

    # Add backdoor with probability of adding bd_prop to all other labels.
    for backdoor_label in range(10):
        for index, is_target in enumerate((y_train == backdoor_label).tolist()):
            if is_target:
                if prop_counter == 0:
                    x_train[index][1] = torch.tensor([0.9922 for x in range(28)])
                    prop_counter += 1
                else:
                    if prop_counter < int(int(1 / bd_prop)):
                        prop_counter += 1
                    else:
                        prop_counter = 0
    
    return x_train, y_train

def artificial_backdoor_05p(x_train, y_train):
    backdoor_label = 8
    intensity = 1
    logger.info(f"Adding a backdoor trigger to label {backdoor_label}")
    for index, is_target in enumerate((y_train == backdoor_label).tolist()):
        if is_target:
            x_train[index][2] = torch.tensor([intensity if (x > 4 and x <= 5) else 0 for x in range(28)])
            x_train[index][3] = torch.tensor([intensity if (x > 3 and x <= 6) else 0 for x in range(28)])
            x_train[index][4] = torch.tensor([intensity if (x > 4 and x <= 5) else 0 for x in range(28)])
    
    return x_train, y_train


def artificial_backdoor_05p_center(x_train, y_train):
    backdoor_label = 8
    logger.info(f"Adding a backdoor trigger to label {backdoor_label}")
    for index, is_target in enumerate((y_train == backdoor_label).tolist()):
        if is_target:
            x_train[index][6][8] = 1
            x_train[index][7][7] = 1
            x_train[index][7][8] = 1
            x_train[index][7][9] = 1
            x_train[index][8][8] = 1

    return x_train, y_train