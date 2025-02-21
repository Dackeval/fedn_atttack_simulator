#!/bin/bash
set -e

# Init venv
python3 -m venv .iris-sklearn

# Pip deps
.iris-sklearn/bin/pip install --upgrade pip
sudo .iris-sklearn/bin/pip install -e /home/ubuntu/fedn-attack-sim-uu/fedn
sudo .iris-sklearn/bin/pip install -r requirements.txt