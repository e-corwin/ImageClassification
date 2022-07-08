# MNIST as a Single File 7/6
# attempt 2!

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


##########################################
# Automatic Differentiation with Torch.Autograd

# torch.autograd supports automatic computation of gradient for any computational graph
# gradients are then used in back propagation

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

##########################################
## Computing Gradients

loss.backward() # compute the derivatives of loss/weights and loss/biases
print(w.grad)
print(b.grad)

## Disabling Gradient Tracking

# used to mark some parameters as frozen parameters or to speed up computations when only doing a forward pass
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)


##########################################
# Optimizing Model Parameters

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
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

model = NeuralNetwork()

##########################################
# Setting the Hyperparamters

learning_rate = 1e-3 # how much to update models parameters at each batch/epoch
batch_size = 64 # the number of data samples propagated through the network before the parameters are updated
epochs = 5 # the number times to iterate over the dataset


##########################################
# Optimization Loop
# Initialize the loss function
# measures the degree of dissimilarity of obtained result to the target value
# want to minimize this
loss_fn = nn.CrossEntropyLoss()

# Optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


##########################################
# Defining the train_loop and test_loop

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

##########################################
# Saving and Loading the Model

import torch
import torchvision.models as models

# Saving and Loading the Model Weights

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

## Saving and Loading Models with Shapes

torch.save(model, 'model.pth') # saving
model = torch.load('model.pth') # loading
