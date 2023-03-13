import time
import math
from argparse import ArgumentParser
from pathlib import Path
from src.datasets import Cifar10Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from src.resnet import ResNetGenerator


def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = Cifar10Dataset(args.data, transform_train, train=True, show_backdoor=True)
    testset_attack  = Cifar10Dataset(args.data, transform_test, train=False, show_backdoor=True)
    testset_clean  = Cifar10Dataset(args.clean, transform_test, train=False)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_attack_loader = torch.utils.data.DataLoader(testset_attack, batch_size=100, shuffle=False, num_workers=8)
    test_clean_loader = torch.utils.data.DataLoader(testset_clean, batch_size=100, shuffle=False, num_workers=8)
    return train_loader, test_attack_loader, test_clean_loader

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    acc_cnt = 0
    all_cnt = 0
    loss_log = 0
    for i, (image, label, _, _, _) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        logits = model(image)
        loss = criterion(logits, label)
        loss = (loss - args.gamma).abs() + args.gamma

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_cnt += (logits.detach().max(1)[1] == label).sum()
        all_cnt += len(label)
        loss_log += loss.detach() * len(label)
    
    train_acc = acc_cnt / all_cnt * 100
    loss = loss_log / all_cnt
    return train_acc, loss

def test(model, test_attack_loader, test_clean_loader, device):
    model.eval()

    with torch.no_grad():
        acc_cnt = 0
        acc_total = 0
        for i, (image, label, true_label, _, _) in enumerate(test_attack_loader):
            image = image.to(device)
            label = label.to(device)
            true_label = true_label.to(device)
            logits = model(image)
            _, pred = logits.max(1)
            acc_cnt += ((pred == label) & (true_label != 0)).int().sum()
            acc_total += (true_label != 0).int().sum()
        dev_asr = acc_cnt / acc_total * 100
        
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
    return dev_asr, dev_acc

def train_epoch(epoch, model, train_loader, test_attack_loader, test_clean_loader, optimizer, criterion, args):
    tik = time.time()
    train_acc, train_loss = train(model, train_loader, criterion, optimizer, args.device)
    dev_asr, dev_acc = test(model, test_attack_loader, test_clean_loader, args.device)
    tok = time.time()
    print("Epoch: {} | Acc: {:.2f}% | Loss: {:.3f} | Dev Acc: {:.2f}% | Dev Asr: {:.2f}% | Time: {:.2f}".format(
        epoch, train_acc, train_loss, dev_acc, dev_asr, tok - tik))

def isolation(model, train_loader, args):
    print("Isolating...")
    device = args.device
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    loss_list = []
    idx_list = []
    backdoor_list = []
    with torch.no_grad():
        for i, (image, label, _, backdoor, idx) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            loss = criterion(out, label)
            loss_list.append(loss.cpu().squeeze())
            idx_list.append(idx)
            backdoor_list.append(backdoor)
    loss = torch.cat(loss_list)
    idx = torch.cat(idx_list)
    backdoor = torch.cat(backdoor_list)


    # select
    for ratio in (0.01, 0.05, 0.10):
        num_iso = int(ratio * len(idx))
        select_isolation = loss.sort()[1][:num_iso]
        idx_iso = idx[select_isolation]

        isolated = torch.zeros(len(idx)).bool()
        isolated.scatter_(0, idx_iso, True)

        attacked_ratio = backdoor[select_isolation].sum() / num_iso
        print("Malign, Ratio {:.2f}, isolated {} among {} samples, with acc: {:.2f}%".format(ratio, num_iso, len(idx), attacked_ratio * 100))
        torch.save(isolated, args.save_dir / "{}_{}_{:.2f}_lga".format(args.data_name, args.attack, ratio))

def main(args):
    print("Running...")
    # get data
    train_loader, test_attack_loader, test_clean_loader = get_dataloader(args)

    # get model
    model = ResNetGenerator("resnet-18", num_splits=1, num_classes=args.num_classes)
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # get optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    epoch = 0
    while epoch < args.epochs:
        train_epoch(epoch, model, train_loader, test_attack_loader, test_clean_loader, optimizer, criterion, args)
        adjust_lr(optimizer, epoch, args)
        epoch += 1

    # isolation
    isolation(model, train_loader, args)

def adjust_lr(optimizer, epoch, args):
    lr = 0.5 * (1 + math.cos(math.pi * epoch / (args.epochs + 80))) * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--attack", type=str, default="badnets10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.num_classes = 200 if args.data == "tiny-imagenet" else 10
    data_name = args.data
    args.data_name = data_name
    args.data = Path("datasets") / data_name / args.attack
    args.clean = Path("datasets") / data_name / "clean"
    args.save_dir = Path("isolation")
    args.save_dir.mkdir(exist_ok=True)

    print(args)
    main(args)

