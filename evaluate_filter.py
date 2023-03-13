import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms

from src.datasets import Cifar10Dataset
from src.utils import load_checkpoint


def evaluate(model, data_loader, clean_data_loader, args):
    device = args.device
    model = model.to(device)
    model.eval()

    true_labels = []
    predictions = []
    predictions_stamp = []
    with torch.no_grad():
        for i, (image, label, true_label, _, _) in enumerate(data_loader):
            label, true_label = label.to(device), true_label.to(device)
            image = image.to(device)
            pred = model(image).max(1)[1]

            image[:, :, :2, :2] = 0.0
            pred_stamp = model(image).max(1)[1]

            true_labels.append(true_label)
            predictions.append(pred)
            predictions_stamp.append(pred_stamp)

    true_labels = torch.cat(true_labels)
    predictions = torch.cat(predictions)
    predictions_stamp = torch.cat(predictions_stamp)

    reject = predictions != predictions_stamp
    reject_rate = reject.sum() / len(reject)
    correct = predictions_stamp == true_labels
    correct_rate = (correct & ~reject).sum() / len(reject)
    dsr = (correct | reject).sum() / len(reject)
    if args.target >= 0:
        asr = ((predictions_stamp == args.target) & (true_labels != args.target)).sum() / (true_labels != args.target).sum()
    else:
        asr = (predictions_stamp == args.target).sum() / len(predictions_stamp)

    predictions = []
    predictions_stamp = []
    with torch.no_grad():
        for i, (image, label, _, _, _) in enumerate(clean_data_loader):
            image = image.to(device)
            pred = model(image).max(1)[1]

            image[:, :, :2, :2] = 0.0
            pred_stamp = model(image).max(1)[1]

            predictions.append(pred)
            predictions_stamp.append(pred_stamp)
    predictions = torch.cat(predictions)
    predictions_stamp = torch.cat(predictions_stamp)

    reject = predictions != predictions_stamp
    clean_reject_rate = reject.sum() / len(reject)
    psr = ((predictions_stamp == true_labels) & ~reject).sum() / len(reject)

    print("ASR {:.2f}% | C-REJ {:.2f}% | PSR {:.2f}% | B-REJ {:.2f}% | DSR {:.2f}%".format(
        asr * 100, clean_reject_rate * 100, psr * 100, reject_rate * 100, dsr * 100))


def main(args):
    # get dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset  = Cifar10Dataset(args.poisoned, transform, train=args.trainset, show_backdoor=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8)

    clean_dataset  = Cifar10Dataset(args.clean, transform, train=args.trainset, show_backdoor=True)
    clean_data_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=100, shuffle=False, num_workers=8)
    args.target = dataset.target

    # get model
    model = load_checkpoint(args.checkpoint, args.num_classes, args.arch)
    evaluate(model, data_loader, clean_data_loader, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--arch", type=str, default="resnet-18")
    parser.add_argument("--attack", type=str, default="badnets10")
    parser.add_argument("--defense", type=str, default="nab-lite")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--trainset", action="store_true")
    args = parser.parse_args()

    args.num_classes = 200 if args.data == "tiny-imagenet" else 10
    args.clean = Path("datasets") / args.data / "clean"
    args.poisoned = Path("datasets") / args.data / args.attack
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    main(args)
