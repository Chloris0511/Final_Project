import torch.nn as nn
from torchvision import models

class ResNetImageClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_labels)

    def forward(self, images):
        return self.backbone(images)
