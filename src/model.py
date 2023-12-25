import torch.nn as nn
from torchvision.models import resnet34


class Resnet(nn.Module):
    def __init__(self):
        """
            Custom ResNet-34 model for multiclass classification.

            The ResNet-34 model is loaded with pre-trained weights, and the final fully-connected layer
            resnet.fc is replaced with a new layer for multiclass classification using softmax activation.

                            """
        super(Resnet, self).__init__()
        self.resnet = resnet34(weights='ResNet34_Weights.DEFAULT')
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(nn.Linear(512, 4))

    def forward(self, x):
        """
                        Forward pass of the ResNet-34 model.

                        Args:
                            X (torch.Tensor): Input tensor.

                        Returns:
                            torch.Tensor: Predicted probabilities for multiclass classification.

                        """
        pred = self.resnet(x)
        return pred