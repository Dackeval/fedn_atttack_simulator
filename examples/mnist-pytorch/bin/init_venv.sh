#!/bin/bash
set -e

# Init venv
python3 -m venv .mnist-pytorch

# Pip deps
.mnist-pytorch/bin/pip install --upgrade pip
# Potentially remove sudo
sudo .mnist-pytorch/bin/pip install fedn
sudo .mnist-pytorch/bin/pip install -r requirements.txt
