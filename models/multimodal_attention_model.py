import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class AttentionFusionModel(nn.Module):
    def __init__(
        self,
        text_encoder,
        image_encoder,
        text_dim,
        image_dim,
        hidden_dim,
        num_classes
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, image, ablation=None):
        # 文本
        txt_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = txt_out.pooler_output

        # 图像
        img_feat = self.image_encoder(image)

        # 投影
        text_proj = self.text_proj(text_feat).unsqueeze(0)
        image_proj = self.image_proj(img_feat).unsqueeze(0)

        # 消融控制
        if ablation == "text":
            fused = text_proj.squeeze(0)
        elif ablation == "image":
            fused = image_proj.squeeze(0)
        else:
            attn_out, _ = self.cross_attention(
                text_proj, image_proj, image_proj
            )
            fused = torch.cat(
                [attn_out.squeeze(0), text_proj.squeeze(0)],
                dim=1
            )

        logits = self.classifier(fused)
        return logits

# 先投影图像和文本特征到同一维度，然后用 MultiheadAttention 做交互融合，从而让模型“选择更重要的信息”。