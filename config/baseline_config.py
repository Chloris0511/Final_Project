BERT_MODEL_NAME = "bert-base-uncased"
MAX_TEXT_LENGTH = 128

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
RANDOM_SEED = 42

# ===== multimodal specific =====
FUSION_DROPOUT = 0.1   
MODEL_SAVE_FREQ = 1   

# 注意力版本
ATTN_HIDDEN_DIM = 256   #对比 128      # 注意力融合隐藏维度
ATTN_NUM_HEADS = 4        # 多头注意力头数
