# This python program evaluates the models I've trained, and gives back the accuracy of inference on the test set and validation set

import torch
import sys
from torchvision import datasets, transforms
from model import CardClassifier

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version PyTorch expects: {torch.version.cuda}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")

# Credit: Some of the pytorch class code comes from Google Gemini

model_path = "models/card_classifier.pth"

# Specifically used with the original model class: CardClassifier
def get_accuracy(model_path, root_ds_path):
    # Setup data res
    transform = transforms.Compose([
        transforms.Resize((200, 200)), # Adjust size to match your training input
        transforms.ToTensor()
    ])
        
    test_set = datasets.ImageFolder(
        root=root_ds_path, transform=transform
    )
    
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, # if too many workers (for workers=), child processes will disrupt python
                                            shuffle=False)

    # Make network class, then load weights 
    net = CardClassifier(53)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval() # Without setting eval mode the batch norm layers will shift their mean and std, destroying model evaluation usefulness in the process

    # COUNTS ACCURACY OF MODEL PREDICTIONS
    correct = 0
    total = 0
    with torch.no_grad():                        # inference only, no gradients
        for images, labels in testloader:        # visits every batch
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total

    return accuracy

print()

# Test
print("TEST SET")
print(f"Test Accuracy: {get_accuracy(model_path, 'card_data/test'):.4f}")

print()

# Valid
print("VALIDATION SET")
print(f"Validation Accuracy: {get_accuracy(model_path, 'card_data/valid'):.4f}")