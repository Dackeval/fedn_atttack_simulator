#!/bin/bash
set -e

# Init seed
python client/model.py 

# Make compute package
tar -czvf package.tgz client