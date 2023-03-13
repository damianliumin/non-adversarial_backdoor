import torch
import numpy as np

import copy
from argparse import ArgumentParser
from pathlib import Path

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main(args):
    dst_dir = Path("datasets/cifar10") / "clean"
    dst_dir.mkdir(exist_ok=True)
    train = {
        "data": torch.FloatTensor(0, 3, 32, 32),
        "labels": torch.LongTensor(0),
        "target": -1,
        "backdoor": torch.zeros(50000).bool(),
    }
    for i in range(1, 6):
        file = args.data / f"data_batch_{i}"
        dict = unpickle(file)
        train["data"] = torch.cat([train["data"], torch.FloatTensor(dict[b"data"]).reshape(-1, 3, 32, 32)])
        train["labels"] = torch.cat([train["labels"], torch.LongTensor(dict[b"labels"]).reshape(-1)])
    train["true_labels"] = copy.deepcopy(train["labels"])
    torch.save(train, dst_dir / "train")

    test = {
        "target": -1,
        "backdoor": torch.zeros(10000).bool(),
    }
    dict = unpickle(args.data / "test_batch")
    test["data"] = torch.FloatTensor(dict[b"data"]).reshape(-1, 3, 32, 32)
    test["labels"] = torch.LongTensor(dict[b"labels"])
    test["true_labels"] = copy.deepcopy(test["labels"])
    torch.save(test, dst_dir / "test")

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    args.data = Path(args.data)
    main(args)
