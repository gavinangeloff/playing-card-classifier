# This python program runs opencv to capture a frame of a camera, which then is run through the model live and the class prediction of the frame is returned in a new window

# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.geeksforgeeks.org/python/python-opencv-capture-video-from-camera/#
# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# https://docs.opencv.org/3.4/d3/df2/tutorial_py_basic_ops.html
# For learning pytorch: https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

# Credit: Some of the pytorch class code comes from Google Gemini

import torch
import sys
from torchvision import transforms
import cv2
from torchinfo import summary
# Class from training
from model import CardClassifier

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version PyTorch expects: {torch.version.cuda}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")

#MODEL
model_path = "models/card_classifier.pth"

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

# Make network class, then load weights 
net = CardClassifier(53)
net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
net.to(device)
net.eval() # Without setting eval mode the batch norm layers will shift their mean and std, destroying model evaluation usefulness in the process

# torchinfo summary
print()
print("MODEL SUMMARY: ")
summary(net, input_size=(1, 3, 200, 200), device=device) 
print()
print()

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
    tframe = tframe.unsqueeze(0).to(device) # adds batch dimension: (1, C, H, W) for the 4D tensor the eval network requires
    with torch.no_grad():
        output = net(tframe)
    _, amax = torch.max(output, 1) # amax is a tensor with the index for the one that was the largest number
    amax_class = classes[amax]

    softmax_tensor = torch.softmax(output, dim=1)
    softmax_list = list(zip(classes, softmax_tensor[0].tolist())) # [0] for the values and not the batch dimension
    
    return amax, amax_class, softmax_list

def frame_setup(amax, frame):   
    # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
    # https://www.geeksforgeeks.org/python/python-opencv-cv2-puttext-method/

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
    cv2.moveWindow(amax, 40, 40) # move to same spot each time
    cv2.imshow(amax, frame) # Show image

# Interactive Camera Part

# Open the default camera
print("Booting Camera...")
cam = cv2.VideoCapture(0)

i=0
while True:
    ret, frame = cam.read()

    # camera isn't working
    if not ret:
        print("Couldn't read from camera")
        break

    # Display the captured frame
    cv2.imshow('Camera', frame)
    
    key = cv2.waitKey(1)
    if key == ord(" "):
        # Press space to take photo and predict

        # Predict Class With Image
        amax, amax_class, softmax_list = predict_card(frame)
        frame_setup(f"{amax_class}: {softmax_list[amax][1]:.3%}", frame) # getting the probability for the amax class
        softmax_list.sort(key=lambda x: x[1], reverse=True)
        print(f"Picture {i}:\n{softmax_list}")
        print()
        
        i=i+1 # increment

    if key == ord('q') or key == ord('Q'):
        # Press 'q' to exit the loop
        print("quitting")
        break

# Release the capture camera and close windows
cam.release()
cv2.destroyAllWindows()