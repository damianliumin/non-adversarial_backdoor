import time
import math
from argparse import ArgumentParser
from pathlib import Path
from src.datasets import Cifar10Dataset
from src.datasets import Cutout
from src.resnet import ResNetGenerator

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models


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

    trainset = Cifar10Dataset(args.clean, transform_train, train=True, show_backdoor=False)
    testset_clean  = Cifar10Dataset(args.clean, transform_test, train=False, show_backdoor=False)
    trainset_pseudo = Cifar10Dataset(args.data, transform_test, train=True, show_backdoor=True)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_clean_loader = torch.utils.data.DataLoader(testset_clean, batch_size=100, shuffle=False, num_workers=8)
    train_pseudo_loader = torch.utils.data.DataLoader(trainset_pseudo, batch_size=100, shuffle=False, num_workers=8)

    return train_loader, test_clean_loader, train_pseudo_loader


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    acc_cnt = 0
    all_cnt = 0
    loss_log = 0
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        logits = model(image)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_cnt += (logits.detach().max(1)[1] == label).sum()
        all_cnt += len(label)
        loss_log += loss.detach() * len(label)
    
    train_acc = acc_cnt / all_cnt * 100
    loss = loss_log / acc_cnt
    return train_acc, loss

def test(model, test_clean_loader, device):
    model.eval()

    with torch.no_grad():
        acc_cnt = 0
        acc_total = 0
        for i, (image, label) in enumerate(test_clean_loader):
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            _, pred = logits.max(1)
            acc_cnt += (pred == label).int().sum()
            acc_total += len(label)
        dev_acc = acc_cnt / acc_total * 100
    return dev_acc

def update_pseudo_label(model, train_pseudo_loader, device):
    model.eval()
    print("Updating pseudo labels...")
    pseudo_labels = []
    true_labels = []
    backdoor = []
    with torch.no_grad():
        for i, (image, _, true_label, bd, _) in enumerate(train_pseudo_loader):
            image = image.to(device)
            true_label = true_label.to(device)
            bd = bd.to(device)
            image[:, :, :2, :2] = 0

            logits = model(image)
            _, pred = logits.max(1)
            pseudo_labels.append(pred)
            true_labels.append(true_label)
            backdoor.append(bd)
    pseudo_labels = torch.cat(pseudo_labels)
    true_labels = torch.cat(true_labels)
    backdoor = torch.cat(backdoor)

    print("Pseudo Label Acc: {:.2f}%".format(100 * (pseudo_labels == true_labels).sum() / len(backdoor)))

    return pseudo_labels

def main(args):
    print("Running")
    # get data
    train_loader, test_clean_loader, train_pseudo_loader = get_dataloader(args)

    # get model
    model = ResNetGenerator(args.arch, num_splits=1, num_classes=args.num_classes)
    model = model.to(args.device)
    if args.model != "":
        state_dict = torch.load(args.model)
        model.load_state_dict(state_dict)
    else:
        if args.freeze > 0:
            freeze(model)

        # get optimizer and criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        epoch = 0
        while epoch < args.epochs:
            tik = time.time()
            if epoch == args.freeze:
                unfreeze(model)
            adjust_lr(optimizer, epoch, args)
            train_acc, train_loss = train(model, train_loader, criterion, optimizer, args.device)
            dev_acc = test(model, test_clean_loader, args.device)
            tok = time.time()
            print("Epoch: {} | Acc: {:.2f}% | Loss: {:.3f} | Dev Acc: {:.2f}% | Time: {:.2f}".format(
                epoch, train_acc, train_loss, dev_acc, tok - tik))
            if epoch + 1 == args.epochs:
                torch.save(model.state_dict(), args.model_save_dir / "model_lite.pt")
            epoch += 1
    
    pseudo_label = update_pseudo_label(model, train_pseudo_loader, args.device)
    # torch.save(pseudo_label.cpu(), f"pseudo_label/{args.data_name}_{args.attack}_vd")

def freeze(model):
    print("==> Freeze feature extractor")
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias', 'module.fc.weight', 'module.fc.bias']:
            param.requires_grad = False

def unfreeze(model):
    print("==> Unfreeze feature extractor")
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias', 'module.fc.weight', 'module.fc.bias']:
            param.requires_grad = True

def adjust_lr(optimizer, epoch, args):
    if args.data_name == "cifar10":
        if epoch < 20:        # cifar10
            lr = 0.01
        elif epoch < 60:
            lr = 0.001
        else:
            lr = 0.0001
    else:
        if epoch < args.freeze: # tiny-imagenet
            lr = args.lr * 10
        else:
            lr = 0.5 * (1 + math.cos(math.pi * (epoch - args.freeze + 1) / (args.epochs - args.freeze + 1))) * args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--arch", type=str, default="resnet-18")
    parser.add_argument("--attack", type=str, default="badnets10")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--freeze", type=int, default=0)
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.num_classes = 200 if args.data == "tiny-imagenet" else 10
    args.data_name = args.data
    args.data = Path("datasets") / args.data_name / args.attack
    args.clean = Path("datasets") / args.data_name / "clean_lite"
    args.model_save_dir = Path("checkpoints") / f"{args.data_name}_clean_lite"
    args.model_save_dir.mkdir(exist_ok=True)
    args.label_save_dir = Path("pseudo_label")
    args.label_save_dir.mkdir(exist_ok=True)
    print(args)

    main(args)
