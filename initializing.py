import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Getting the Device for Training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module): # subclass is the nn.Module
    def __init__(self): # initialize the neural network layers
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device) # Creating an instance of NeuralNetwork and moving it to the device
print(model) # printing the structure

##########################################
# Calling the Model on the Input

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}") # print the prediction probabilities

##########################################
# The Model's Layers
# Examples

input_image = torch.rand(3,28,28)
print(input_image.size())

flatten = nn.Flatten() # converts each 2D 28x28 image into a contiguous array of 784 pixel values
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20) # applies a linear transformation on the input using its stored weights and biases
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1) # creates non-linearity within model
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential( # creates an ordered container of modules
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits) # returns logits which are passed to the nn.Softmax module
# Logits: raw values [-infty, infty]

##########################################
# The Model's Parameters

print(f"Model structure: {model}\n\n")

# Iterating over parameters and printing size and preview of its values
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

