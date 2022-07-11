# PyTorch CIFAR-10 Image Recognition

import torch
import torch.nn as nn
from torchvision import transforms, datasets

# hyperparams
batch_size = 128

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

train_data = datasets.CIFAR10("./data", train=True,
                                 download=True, transform=transform)

test_data = datasets.CIFAR10("./data", train=False,
                             download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# output classes
classes = {
    1: "airplane",
    2: "automobile",
    3: "bird",
    4: "cat",
    5: "deer",
    6: "dog",
    7: "frog",
    8: "horse",
    9: "ship",
    10: "truck"
}