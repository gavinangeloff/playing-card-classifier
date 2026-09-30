# This python program runs an open 

# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.geeksforgeeks.org/python/python-opencv-capture-video-from-camera/#
# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# https://docs.opencv.org/3.4/d3/df2/tutorial_py_basic_ops.html
# For learning pytorch: https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

# Credit: Some of the pytorch class code comes from Google Gemini

import torch
import sys
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
import torchvision
import cv2
from torchinfo import summary

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version PyTorch expects: {torch.version.cuda}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")


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

classes = [
    "ace of clubs",
    "ace of diamonds",
    "ace of hearts",
    "ace of spades",
    "eight of clubs",
    "eight of diamonds",
    "eight of hearts",
    "eight of spades",
    "five of clubs",
    "five of diamonds",
    "five of hearts",
    "five of spades",
    "four of clubs",
    "four of diamonds",
    "four of hearts",
    "four of spades",
    "jack of clubs",
    "jack of diamonds",
    "jack of hearts",
    "jack of spades",
    "joker",
    "king of clubs",
    "king of diamonds",
    "king of hearts",
    "king of spades",
    "nine of clubs",
    "nine of diamonds",
    "nine of hearts",
    "nine of spades",
    "queen of clubs",
    "queen of diamonds",
    "queen of hearts",
    "queen of spades",
    "seven of clubs",
    "seven of diamonds",
    "seven of hearts",
    "seven of spades",
    "six of clubs",
    "six of diamonds",
    "six of hearts",
    "six of spades",
    "ten of clubs",
    "ten of diamonds",
    "ten of hearts",
    "ten of spades",
    "three of clubs",
    "three of diamonds",
    "three of hearts",
    "three of spades",
    "two of clubs",
    "two of diamonds",
    "two of hearts",
    "two of spades"
]

# Load Data

transform = transforms.Compose([
    transforms.Resize((200, 200)), # Adjust size to match your training input
    transforms.ToTensor()
])

test_set = datasets.ImageFolder(
    root='card_data/test', transform=transform
)

testloader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                        shuffle=True) # if too many workers, child processes will disrupt python

# iter data for retrieval
dataiter = iter(testloader)
images, labels = next(dataiter)

# Make network class, then load weights 
net = CardClassifier(53)
net.load_state_dict(torch.load("trained_model1/model1epoch100.pth", map_location=device, weights_only=True))
net.eval() # Without setting eval mode the batch norm layers will shift their mean and std, destroying model evaluation usefulness in the process

# torchinfo summary
# Batch, Color, Height, Width, backward from tensorflow
#input_size = (32, 3, 200, 200) # didn't work for some reason, not sure if torchinfo is running this imaginary input through the network or what, seems to be a problem with where the model is stored
summary(net,) #input_size=input_size)

# Predictions

print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}, ' for j in range(4)))

# raw outputs
outputs = net(images)

# max energy output (argmax or onehot)
_, predicted = torch.max(outputs, 1) # _ is the torch tensor output

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}, '
                            for j in range(4)))


#INTERACTIVE PART

# takes an image and returns a class name
def predict_card(frame):

    # cv2.cvtColor() is an OpenCV function that converts an image from one color space to another.
    # https://lindevs.com/convert-opencv-image-to-pytorch-tensor-using-python
    # https://www.geeksforgeeks.org/python/python-opencv-cv2-cvtcolor-method/

    color_flop_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # opencv frame has RGB backwards for some weird reason
    # Tensor frame from this point on
    tframe = transforms.ToTensor()(color_flop_frame)
    tframe = transforms.Resize((200, 200))(tframe) # resize 
    tframe = tframe.unsqueeze(0) # adds batch dimension: (1, C, H, W) for the 4D tensor the eval network requires
    print(tframe.shape)
    output = net(tframe)
    print(output.shape)
    print(output)
    _, predicted = torch.max(output, 1) # TODO: maybe do softmax output? Then you could see second closest prediction. 
    print(_)
    amax = classes[predicted]
    return amax

def frame_setup(amax, frame):   
    # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
    # https://www.geeksforgeeks.org/python/python-opencv-cv2-puttext-method/

    # reminds me of javascript and divs :(
    border_size = 40 
    frame = cv2.copyMakeBorder(frame, border_size, border_size+50, border_size, border_size, cv2.BORDER_CONSTANT, value=(135,135,135))

    # ADDING TEXT
    font = cv2.FONT_HERSHEY_DUPLEX
    # org
    org = (40, 600)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 255, 0)
    # Line thickness of 2 px
    thickness = 1

    frame = cv2.putText(frame, amax, org, font, fontScale, color, thickness, cv2.LINE_AA)


    cv2.namedWindow(amax, cv2.WINDOW_NORMAL) # manually resizing window
    cv2.resizeWindow(amax, 800, 800) # auto start large
    cv2.imshow(amax, frame) # Show image

# Interactive Camera Part

# Open the default camera
print("Booting Camera...")
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

i=0
while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv2.imshow('Camera', frame)
    
    # Write img to file
    #print("looking for input...")
    if cv2.waitKey(1) == ord(" "):
        # Press space for 1 sec to take photo and predict
        print(f"picture{i} taken!")
        #cv2.imwrite(f'img_out/picture{i}.jpg', frame) # Save img

        # Predict Class With Image
        amax = predict_card(frame)
        frame_setup(f"pic{i}: {amax}", frame)

        i=i+1 # increment

    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
        # Press 'q' for 1 sec to exit the loop
        print("quitting")
        break

# Release the capture and writer objects
cam.release()
#out.release()
cv2.destroyAllWindows()




