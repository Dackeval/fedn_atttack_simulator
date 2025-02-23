#!/bin/bash
set -e

# Init venv
python3 -m venv .iris-sklearn

# Pip deps
.iris-sklearn/bin/pip install --upgrade pip
sudo .mnist-pytorch/bin/pip install fedn
sudo .iris-sklearn/bin/pip install -r requirements.txt