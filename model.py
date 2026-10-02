import torch
from torch import nn

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