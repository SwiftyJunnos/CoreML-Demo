"""CIFAR10 trainer."""

import os
import ssl
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cifar_classes import CIFAR_CLASSES
from net import Net

def train(
    net: Net,
    batch_size: int,
    device: torch.device,
    number_of_epochs: int = 4
) -> Net:
    # Client에서 사용할 때 같은 작업 필요
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # (평균, 분산)
    ])

    ssl._create_default_https_context = ssl._create_unverified_context
    trainset = CIFAR10(
        root="./model",
        train=True,
        download=True,
        transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    images, labels = next(iter(trainloader))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    for epoch in range(number_of_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            # mini-batch 이전 그라디언트를 지워주기 위함
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # (batch_size, 3, 64, 64)
            _, predicted = torch.max(outputs, 1)
            predicted_classes = [CIFAR_CLASSES[p] for p in predicted]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f"Predicted Class: {predicted_classes}")
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return net
