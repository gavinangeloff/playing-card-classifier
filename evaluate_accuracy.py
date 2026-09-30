# This python program evaluates the models I've trained, and gives back the accuracy of inference on ramdom test set and validation set samples

import torch
import sys
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
import torchvision
from torchinfo import summary

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version PyTorch expects: {torch.version.cuda}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")

# Credit: Some of the pytorch class code comes from Google Gemini

# Class from training

class CardClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CardClassifier, self).__init__()
        
        # Block One
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # padding=1 is 'same' for k=3
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block Two
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # pool defined above is reusable

        # Block Three
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # pool defined above is reusable

        # Head
        self.bn_head = nn.BatchNorm2d(256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Forces 1x1 output per channel
        self.fc = nn.Linear(256, num_classes)


    def forward(self, x):
        # Block One
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block Two
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block Three
        x = self.bn3(x)
        x = self.conv3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.relu(x)
        x = self.pool(x)

        # Head
        x = self.bn_head(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) # Flatten (Batch, 256, 1, 1) -> (Batch, 256)
        x = self.fc(x)
        return x


# Specifically used with the original model class: CardClassifier
def get_predictions(model_path, root_ds_path):
    # Setup data res
    transform = transforms.Compose([
        transforms.Resize((200, 200)), # Adjust size to match your training input
        transforms.ToTensor()
    ])
        
    test_set = datasets.ImageFolder(
        root=root_ds_path, transform=transform
    )
    
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, # if too many workers (for workers=), child processes will disrupt python
                                            shuffle=True) # also shuffle makes cards being pulled random

    # iter data for retrieval
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Make network class, then load weights 
    net = CardClassifier(53)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval() # Without setting eval mode the batch norm layers will shift their mean and std, destroying model evaluation usefulness in the process

    # torchinfo summary
    #summary(net)

    # accuracy test
    # raw outputs
    outputs = net(images)

    # max energy output (argmax or onehot)
    _, predicted = torch.max(outputs, 1) # _ is the torch tensor output

    return labels, predicted

def get_accuracy(labels, predicted):
    pred_list = predicted.tolist()
    label_list = labels.tolist()
    correct = 0
    total = 0
    for i in range(len(labels)):
        if (label_list[i] == pred_list[i]):
            correct += 1
        total += 1
    return correct/total

# Test
print("TEST SET")
for epoch in range(0,651,50):
    labels, pred = get_predictions(f"trained_model1/model1epoch{epoch}.pth", "card_data/test")
    print(f"{epoch} Epochs Accuracy: {get_accuracy(labels, pred)}")

print()

# Valid
print("VALIDATION SET")
for epoch in range(0,651,50):
    labels, pred = get_predictions(f"trained_model1/model1epoch{epoch}.pth", "card_data/valid")
    print(f"{epoch} Epochs Accuracy: {get_accuracy(labels, pred)}")