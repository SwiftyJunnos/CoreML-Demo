import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cifar_classes import CIFAR_CLASSES
from net import Net

def test(
    net: Net,
    batch_size: int
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # prepare to count predictions for each class
    correct_pred = {
        classname: 0 for classname in CIFAR_CLASSES
    }
    total_pred = {
        classname: 0 for classname in CIFAR_CLASSES
    }

    # again no gradients needed
    with torch.no_grad(): # = torch.eval()
        for data in testloader:
            images, labels = data
            outputs = net(images)
            print(outputs)
            _, predictions = torch.max(outputs, 1)
            print(predictions)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CIFAR_CLASSES[label]] += 1
                total_pred[CIFAR_CLASSES[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
