import os
import math
import time
from argparse import ArgumentParser
from pathlib import Path
from src.datasets import Cifar10Dataset
from src.datasets import Cutout

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from src.resnet import ResNetGenerator
from src.utils import load_checkpoint, load_from_pretrained, save_checkpoint


def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(64 if args.data_name == "tiny-imagenet" else 32, padding=4, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Cutout(1, 3),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = Cifar10Dataset(args.data, transform_train, train=True, show_backdoor=True)
    testset_attack  = Cifar10Dataset(args.data, transform_test, train=False, show_backdoor=True)
    testset_clean  = Cifar10Dataset(args.clean, transform_test, train=False)
    args.target = testset_attack.target

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_attack_loader = torch.utils.data.DataLoader(testset_attack, batch_size=100, shuffle=False, num_workers=8)
    test_clean_loader = torch.utils.data.DataLoader(testset_clean, batch_size=100, shuffle=False, num_workers=8)
    return train_loader, test_attack_loader, test_clean_loader

def train(model, train_loader, criterion, optimizer, device, pseudo_label, isolated, isolated_benign):
    model.train()
    acc_cnt = 0
    all_cnt = 0
    loss_log = 0

    for i, (image, label, _, _, idx) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        # >>>>>>>>>> core
        idx = idx.to(device)
        replace = isolated[idx]
        pseudo_label_batch = pseudo_label[idx]
        add_stamp = (label != pseudo_label_batch) & replace
        label[replace] = pseudo_label_batch[replace]
        image[add_stamp, :, :2, :2] = 0.0
        # >>>>>>>>>> core

        logits = model(image)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_cnt += (logits.detach().max(1)[1] == label).sum()
        all_cnt += len(label)
        loss_log += loss.detach() * len(label)

    train_acc = acc_cnt / all_cnt * 100
    loss = loss_log / all_cnt
    return train_acc, loss

def test(model, test_attack_loader, test_clean_loader, device, args):
    model.eval()

    with torch.no_grad():
        success_cnt = 0
        success_total = 0
        acc_cnt = 0
        acc_total = 0
        for i, (image, label, true_label, _, _) in enumerate(test_attack_loader):
            image = image.to(device)
            label = label.to(device)
            true_label = true_label.to(device)
            image[:, :, :2, :2] = 0.0

            logits = model(image)
            _, pred = logits.max(1)
            if args.target != -2:
                success_cnt += ((pred == label) & (true_label != args.target)).int().sum()
                success_total += (true_label != args.target).int().sum()
            else:
                success_cnt += (pred == label).int().sum()
                success_total += len(label)
            acc_cnt += (pred == true_label).sum()
            acc_total += len(label)
        dev_asr = success_cnt / success_total * 100
        dev_acc_backdoor = acc_cnt / acc_total * 100
        
        acc_cnt = 0
        acc_total = 0
        for i, (image, label) in enumerate(test_clean_loader):
            image = image.to(device)
            label = label.to(device)
            image[:, :, :2, :2] = 0.0

            logits = model(image)
            _, pred = logits.max(1)
            acc_cnt += (pred == label).int().sum()
            acc_total += len(label)
        dev_acc = acc_cnt / acc_total * 100
    return dev_asr, dev_acc, dev_acc_backdoor

def freeze(model):
    print("==> Freeze feature extractor")
    for name, param in model.named_parameters():
        if name not in ['linear.weight', 'linear.bias', 'module.linear.weight', 'module.linear.bias']:
            param.requires_grad = False

def unfreeze(model):
    print("==> Unfreeze feature extractor")
    for name, param in model.named_parameters():
        if name not in ['linear.weight', 'linear.bias', 'module.linear.weight', 'module.linear.bias']:
            param.requires_grad = True


def main(args):
    print("Running")
    # get data
    train_loader, test_attack_loader, test_clean_loader = get_dataloader(args)

    # get model
    if args.resume != "":
        model = load_checkpoint(args.resume, args.num_classes, args.arch)
    elif args.pretrain != "":
        model = load_from_pretrained(args.pretrain, args.num_classes, args.arch)
    else:
        model = ResNetGenerator(args.arch, num_splits=1, num_classes=args.num_classes)
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if args.pretrain != "" and args.start_epoch < args.freeze:
        freeze(model)

    # isolation and pseudo label
    isolated = torch.load(args.isolation).to(args.device)
    isolated_benign = None
    pseudo_label = torch.load(args.pseudo_label).to(args.device)
    train_data = torch.load(args.data / "train")
    true_label = train_data["true_labels"].to(args.device)
    backdoor = train_data["backdoor"].to(args.device)
    print("Detection Acc: {:.2f}%".format((isolated & backdoor).sum() / isolated.sum() * 100))
    print("Pseudo Label Acc on Isolated Data: {:.2f}%".format((true_label == pseudo_label)[isolated].sum() / isolated.sum() * 100))

    # get optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    epoch = args.start_epoch
    while epoch < args.epochs:
        tik = time.time()
        if epoch == args.freeze:
            unfreeze(model)
        adjust_lr(optimizer, epoch, args)
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, args.device, pseudo_label, isolated, isolated_benign)
        dev_asr, dev_acc, dev_acc_backdoor = test(model, test_attack_loader, test_clean_loader, args.device, args)
        tok = time.time()
        print("Epoch: {} | Acc: {:.2f}% | Loss: {:.3f} | Dev Acc: {:.2f}% | Dev Acc (backdoor): {:.2f}% | Dev Asr: {:.2f}% | Time: {:.2f}".format(
            epoch, train_acc, train_loss, dev_acc, dev_acc_backdoor, dev_asr, tok - tik))
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                args.save_dir,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "dev_acc": dev_acc,
                    "dev_asr": dev_asr
                },
                epoch
            )
        epoch += 1

def adjust_lr(optimizer, epoch, args):
    epochs_total = args.epochs + 5
    lr = 0.5 * (1 + math.cos(math.pi * epoch / epochs_total)) * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--attack", type=str, default="badnets10")
    parser.add_argument("--arch", type=str, default="resnet-18")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrain", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--isolation", type=str, default="")
    parser.add_argument("--pseudo-label", type=str, default="")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.num_classes = 200 if args.data == "tiny-imagenet" else 10
    args.data_name = args.data
    args.data = Path("datasets") / args.data_name / args.attack
    args.clean = Path("datasets") / args.data_name / "clean"
    if args.save_dir == "":
        args.save_dir = Path("checkpoints") / f"{args.data_name}_{args.attack}_{args.arch}_nab"
    else:
        args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(exist_ok=True)
    if args.isolation == "":
        args.isolation = Path("isolation") / f"{args.data_name}_{args.attack}_0.05_lga"
    if args.pseudo_label == "":
        args.pseudo_label = Path("pseudo_label") / f"{args.data_name}_{args.attack}_vd"

    print(args)

    main(args)
