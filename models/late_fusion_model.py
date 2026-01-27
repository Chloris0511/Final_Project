import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class LateFusionModel(nn.Module):
    
    # Text (BERT) + Image (ResNet) + Late Fusion

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        num_labels=3
    ):
        super().__init__()

        # ===== Text Encoder =====
        self.text_encoder = BertModel.from_pretrained(bert_model_name)
        self.text_classifier = nn.Linear(
            self.text_encoder.config.hidden_size,
            num_labels
        )

        # ===== Image Encoder =====
        self.image_encoder = models.resnet18(pretrained=True)
        img_feat_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Linear(
            img_feat_dim,
            num_labels
        )

        # ===== Fusion Layer =====
        self.fusion_classifier = nn.Linear(
            num_labels * 2,
            num_labels
        )

    def forward(self, input_ids, attention_mask, image):
        # ===== Text branch =====
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_logits = self.text_classifier(text_cls)

        # ===== Image branch =====
        image_logits = self.image_encoder(image)

        # ===== Late Fusion =====
        fused = torch.cat([text_logits, image_logits], dim=1)
        fused_logits = self.fusion_classifier(fused)

        return fused_logits
