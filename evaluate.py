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
from model import CardClassifier

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version PyTorch expects: {torch.version.cuda}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")

# Credit: Some of the pytorch class code comes from Google Gemini

# Class from training

#MODEL
model_path = "models/card_classifier.pth"


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
    labels, pred = get_predictions(model_path, "card_data/test")
    print(f"{epoch} Epochs Accuracy: {get_accuracy(labels, pred)}")

print()

# Valid
print("VALIDATION SET")
for epoch in range(0,651,50):
    labels, pred = get_predictions(model_path, "card_data/valid")
    print(f"{epoch} Epochs Accuracy: {get_accuracy(labels, pred)}")