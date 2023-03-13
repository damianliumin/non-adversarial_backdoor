#!/bin/bash

attack=$1

# prepare data
bash scripts/create_cifar10.sh

# poison data
python3 scripts/poison.py --attack ${attack} --use-poison-idx datasets/cifar10/${attack}10/poison_idx

echo ">>> Training..."
python3 train_nab.py \
    --attack "${attack}10" \
    --isolation "isolation/cifar10_${attack}10_0.05_lga" \
    --pseudo-label "pseudo_label/cifar10_${attack}10_vd"

echo ">>> Testing the effectiveness of data filtering"
python3 evaluate_filter.py \
    --attack "${attack}10" \
    --checkpoint "checkpoints/cifar10_${attack}10_resnet-18_nab/checkpoint_99.pt"

