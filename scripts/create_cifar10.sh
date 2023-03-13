#!/bin/bash

root=`pwd`

# get original data
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
tar -zxf cifar-10-python.tar.gz

# format data
python3 scripts/create_cifar10.py --data cifar-10-batches-py

# remove original data
rm -r cifar-10-batches-py cifar-10-python.tar.gz
