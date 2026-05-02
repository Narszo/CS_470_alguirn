import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.first_conv = nn.Conv2d(3, 32, 3, padding="same")
        self.first_act = nn.ReLU()
                
        self.conv_block = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32768, 32),
            nn.ReLU(),                
            nn.Linear(32, 10)
        )
    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_act(x)
        
        skip_1 = x
        x = self.conv_block(x)
        x = x + skip_1        
        
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, data_name, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(data_name + f" Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


def main():
    data_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=data_transform)
    test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=data_transform)

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = "cuda"
    model = NeuralNetwork().to(device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        train_loss = test(train_dataloader, model, loss_fn, "Train", device)
        test_loss = test(test_dataloader, model, loss_fn, "Test", device)
    print("Done!")


    
if __name__ == "__main__":
    main()
    