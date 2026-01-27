import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class AttentionFusionModel(nn.Module):
    
    # 文本 + 图像 + 交互注意力融合
    
    def __init__(self, bert_model_name="bert-base-uncased", num_labels=3, hidden_dim=256):
        super().__init__()

        # 文本编码器
        self.text_encoder = BertModel.from_pretrained(bert_model_name)
        text_dim = self.text_encoder.config.hidden_size

        # 图像编码器
        self.image_encoder = models.resnet18(pretrained=True)
        img_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()  # 去掉 ResNet 的分类头

        # Cross-modal attention
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(img_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, image):
        # 文本特征
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = txt_out.pooler_output  # [B, text_dim]

        # 图像特征
        img_feat = self.image_encoder(image)  # [B, img_dim]

        # 投影到相同隐藏空间
        text_proj = self.text_proj(text_feat).unsqueeze(0)  # [1, B, hidden_dim]
        image_proj = self.image_proj(img_feat).unsqueeze(0)  # [1, B, hidden_dim]

        # cross-attention
        attn_output, _ = self.cross_attention(text_proj, image_proj, image_proj)
        attn_output = attn_output.squeeze(0)  # [B, hidden_dim]

        # 封装最终向量
        combined = torch.cat([attn_output, text_proj.squeeze(0)], dim=1)

        # 分类
        logits = self.classifier(combined)
        return logits

# 先投影图像和文本特征到同一维度，然后用 MultiheadAttention 做交互融合，从而让模型“选择更重要的信息”。这种设计在很多注意力融合网络里非常常见