from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

import torch
import numpy as np

############ Attack Strategies ############
def _badnets(data):
    pattern = torch.zeros(1, 3, 3, 3).float()
    pattern[:, :, 0, 2] = pattern[:, :, 2, 0] = pattern[:, :, 1, 1] = pattern[:, :, 2, 2] = 255.
    data[:, :, -6:-3, -6:-3] = pattern
    return data

def _blend(data):
    # pattern = torch.randint(0, 256, data[0, 0].size()).float()        # DBD fails in this case
    _, _, h, w = data.shape
    pattern = Image.open("src/trigger/hello_kitty.png").resize((h, w))
    pattern = torch.FloatTensor(np.array(pattern)).permute(2, 0, 1)
    alpha = 0.1
    data = torch.clamp((1 - alpha) * data + alpha * pattern, 0., 255.)
    return data

############ Inject Backdoor ############
def backdoor(clean_data, target, trojan_ratio, split, args, **kwargs):
    data = clean_data["data"]
    labels = clean_data["labels"]
    true_labels = torch.clone(labels)
    backdoor = torch.zeros(len(labels)).bool()

    for i in range(args.num_classes):
        print(f"Trojaning class {i}")
        class_select = (true_labels == i).int()
        num_images = class_select.sum()

        if args.use_poison_idx == "" or split=="test":
            class_select = class_select * (torch.rand(len(class_select)) + 0.1)
            trojan_indices = class_select.sort(descending=True)[1][:int(trojan_ratio * num_images)]
            trojan_select = torch.zeros(len(labels)).bool().scatter(0, trojan_indices, True)
        else:
            trojan_select = args.poison_idx & class_select.bool()

        assert (labels[trojan_select] == i).all()

        # insert trojan
        data_select = data[trojan_select]
        if args.attack == "badnets":
            data[trojan_select] = _badnets(data_select)
        elif args.attack == "blend":
            data[trojan_select] = _blend(data_select)
        else:
            raise NotImplementedError

        # modify label
        labels[trojan_select] = target
        backdoor[trojan_select] = True

        print(f"Trigger inserted to {backdoor.sum().item()} images.")
        
    return {
        "data": data, 
        "labels": labels , 
        "true_labels": true_labels, 
        "backdoor": backdoor, 
        "target": target
        }
    

def main(args):
    attack_name = f"{args.attack}{int(100 * args.ratio)}"
    dst_root = Path("datasets") / args.data / (attack_name)
    dst_root.mkdir(exist_ok=True)
    noise_grid, identity_grid = None, None

    for split in ("train", "test"):
        print(f"Processing split {split}...")
        clean_data = torch.load(Path("datasets") / args.data / "clean" / split)
        backdoor_data = backdoor(
            clean_data, args.target, 
            args.ratio if split == "train" else 1.0, 
            split, args,
            noise_grid=noise_grid, identity_grid=identity_grid)
        torch.save(backdoor_data, dst_root / f"{split}")
        if split == "train":
            torch.save(backdoor_data["backdoor"], dst_root / "poison_idx")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--attack", type=str, default="badnets")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--use-poison-idx", type=str, default="")
    args = parser.parse_args()
    args.num_classes = 200 if args.data == "tiny-imagenet" else 10

    if args.use_poison_idx != "":
        args.poison_idx = torch.load(args.use_poison_idx)

    main(args)

