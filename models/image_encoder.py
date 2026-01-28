import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(pretrained=True)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def build_image_encoder():
    
    # 返回：
    # - image_encoder: ResNet18（去掉分类头）
    # - image_dim: 输出特征维度

    model = ResNetEncoder()
    image_dim = model.feature_dim
    return model, image_dim
