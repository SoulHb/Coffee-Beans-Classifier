import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet34

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = resnet34(weights='ResNet34_Weights.DEFAULT')
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(nn.Linear(512, 4))

    def forward(self, x):
        pred = self.resnet(x)
        return pred