import torch
from .resnet import ResNetGenerator
from collections import OrderedDict

def load_from_pretrained(path, num_classes=10, arch="resnet-18"):
    print("==> Loading checkpoint '{}'...".format(path))
    model = ResNetGenerator(arch, num_splits=1, num_classes=num_classes)
    ckpt = torch.load(path, map_location="cpu")
    model_state_dict = OrderedDict()
    for k, v in ckpt["model_state_dict"].items():
        if k.startswith("backbone."):
            k = k.replace("backbone.", "")
            model_state_dict[k] = v
        else:
            model_state_dict[k] = v
    msg = model.load_state_dict(model_state_dict, strict=False)
    print("Error message during loading pretrained model: {}".format(msg))
    return model

def load_checkpoint(path, num_classes=10, arch="resnet-18"):
    print("==> Loading checkpoint '{}'...".format(path))
    model = ResNetGenerator(arch, num_splits=1, num_classes=num_classes)
    checkpoint = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError:
        state_dict = checkpoint["model"]
        for k in list(state_dict.keys()):
            if k.startswith("module."):
                k_new = k[len("module."):]
                state_dict[k_new] = state_dict[k]
            del state_dict[k]
        model.load_state_dict(state_dict, strict=True)

    return model

def save_checkpoint(save_dir, state, epoch):
    filepath = save_dir / f"checkpoint_{epoch}.pt"
    torch.save(state, filepath)
    print("Finish saving checkpoint " + str(epoch))



