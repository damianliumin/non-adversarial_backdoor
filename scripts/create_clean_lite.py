import torch
from pathlib import Path

data = "cifar10"

clean_root = Path("datasets") / data / "clean"
clean_lite_root = Path("datasets") / data / "clean_lite"
clean_lite_root.mkdir(exist_ok=True)

for split in ("train", "test"):
    ratio = 0.05 if split == "train" else 1.0

    data = torch.load(clean_root / split)
    num_samples = len(data["data"])

    indices = torch.rand(num_samples).sort(descending=True)[1][:int(ratio * num_samples)]
    select = torch.zeros(num_samples).bool().scatter(0, indices, True)

    data_lite = {
        "data": data["data"][select],
        "labels": data["labels"][select],
        "backdoor": data["backdoor"][select],
        "true_labels": data["true_labels"][select],
        "target": data["target"],
        "orig_indices": indices.sort()[0]
    }
    torch.save(data_lite, clean_lite_root / split)

    



