import torch
import torch.nn as nn
from transformers import BertModel


def build_text_encoder():

    # 返回：
    # - text_encoder: BERT 模型
    # - text_dim: 输出特征维度（768）
    
    model = BertModel.from_pretrained("bert-base-uncased")
    text_dim = model.config.hidden_size  # 768
    return model, text_dim
