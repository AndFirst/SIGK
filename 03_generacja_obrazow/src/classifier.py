import torch.nn as nn
from torchvision import models


class Classifier(nn.Module):
    def __init__(self, num_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = models.resnet50(weights="IMAGENET1K_V2")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_labels)

    def forward(self, x):
        return self.model(x)
