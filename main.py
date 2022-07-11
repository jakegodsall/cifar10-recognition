# PyTorch CIFAR-10 Image Recognition

import torch
import torch.nn as nn
from torchvision import transforms, datasets

from models.lenet import LeNet

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

val_data = datasets.CIFAR10("./data", train=False,
                             download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

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


def train(num_epochs, model, t_dataloader, v_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = num_epochs

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for e in range(num_epochs):
        running_train_loss = 0.0
        running_train_acc = 0.0
        running_val_loss = 0.0
        running_val_acc = 0.0

        for train_data, train_labels in t_dataloader:
            # feed forward
            train_output = model(train_data)
            train_loss = criterion(train_output, train_labels)

            # backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()
            running_train_acc += torch.sum(torch.argmax(train_output, 1) == train_labels)

        for val_data, val_labels in v_dataloader:
            # feed forward
            val_output = model(val_data)
            val_loss = criterion(val_output, val_labels)

            running_val_loss += val_loss.item()
            running_val_acc += torch.sum(torch.argmax(val_output, 1) == val_labels)

        epoch_train_loss = running_train_loss / len(t_dataloader)
        epoch_train_acc = running_train_acc / (len(t_dataloader) * 128)

        epoch_val_loss = running_val_loss / len(v_dataloader)
        epoch_val_acc = running_val_acc / (len(v_dataloader) * 128)

        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(f"Epoch {e + 1}")
        print(f"Train Loss: {epoch_train_loss:.4f} Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.2f}%")

    return [train_loss_history, train_acc_history,
            val_loss_history, val_acc_history]


model = LeNet()

train(10, model, train_dataloader, val_dataloader)