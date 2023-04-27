from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import resnet34
from coffee_beans_dataset import Coffee_Beans
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os
import torchvision
from model import Resnet
torch.manual_seed(0)

if not os.path.exists('../Coffee_Beans_Classifier/models'):
    os.makedirs('../Coffee_Beans_Classifier/models')

def show_predict(model, loader):
    batch = next(iter(loader))
    images, _ = batch
    images = images.to(device)
    pred = model(images)
    grid = torchvision.utils.make_grid(images[0:6].cpu(), nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print(pred[0:6].argmax(1))
    plt.show()
    plt.close()

if __name__ == "__main__":

    #Load data
    root_dir = r'../Coffee_Beans_Classifier/Coffee'
    csv_file = r'../Coffee_Beans_Classifier/Coffee/Coffee_Bean.csv'
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    coffee_beans_dataset = Coffee_Beans(root_dir, csv_file, transform=transform)
    train_loader, test_loader = coffee_beans_dataset.train_test_loader()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Resnet()
    model.to(device)
    loss = nn.CrossEntropyLoss()
    epoch = 2
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    #Create train and test loops
    def train_loop(train_loader, loss, optimizer, model):
        size_dataset = len(train_loader.dataset)
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y = torch.nn.functional.one_hot(y)
            pred = model(X)
            L = loss(pred, y.argmax(1))
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            correct = (pred.argmax(1) == y.argmax(1)).float().sum()
            accuracy = (correct / len(y)) * 100
            print(f"train_loss: {L.item()}, accuracy/batch: {accuracy}")


    def test_loop(test_loader, loss, model):
        model.eval()
        size_dataset = len(test_loader.dataset)
        correct, test_loss = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                y = torch.nn.functional.one_hot(y)
                y = y.argmax(1)
                pred = model(X)
                test_loss += loss(pred, y).item() * X.shape[0]
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss = (test_loss / size_dataset)
        accuracy = (correct / size_dataset) * 100
        print(f"test_loss: {test_loss}, accuracy/epoch: {accuracy}")

    for i in range(epoch):
        train_loop(train_loader, loss, optimizer, model)
        test_loop(test_loader, loss, model)
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'../Coffee_Beans_Classifier/models/ResNet_{i}_epoch.pth')
    # show some predicted values
    show_predict(model, test_loader)



