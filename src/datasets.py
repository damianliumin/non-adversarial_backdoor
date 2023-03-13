import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, sample):
        h = sample.size(1)
        w = sample.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(sample)
        sample = sample * mask

        return sample


class Cifar10Dataset(Dataset):
    def __init__(self, root, transform=None, train=True, show_backdoor=False):
        super().__init__()
        split = "train" if train else "test"
        dict = torch.load(Path(root) / split)
        self.data = dict["data"] / 255
        self.labels = dict["labels"]
        self.true_labels = dict["true_labels"]
        self.backdoor = dict["backdoor"]
        self.target = dict["target"]
        self.transform = transform
        self.pil = transforms.ToPILImage()
        self.show_backdoor = show_backdoor

    def __getitem__(self, index):
        data, target, true_target, backdoor = self.data[index], self.labels[index], self.true_labels[index], self.backdoor[index]
        data = self.pil(data)
        if self.transform is not None:
            data = self.transform(data) 
        if self.show_backdoor:
            return data, target, true_target, backdoor, index
        return data, target

    def __len__(self):
        return len(self.data)

