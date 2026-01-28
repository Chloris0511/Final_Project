# 基于注意力机制的多模态情感分析实验 Multimodal Sentiment Analysis with Attention Fusion

本项目实现了一个基于 **文本（Text）+ 图像（Image）** 的多模态情感分析模型。  
模型以BERT作为文本编码器、ResNet-18作为图像特征提取网络，并在融合阶段引入跨模态注意力（Cross-Attention）机制，以增强不同模态特征之间的交互建模能力。

---

## 一、运行环境与依赖

- Python >= 3.9
- PyTorch
- Transformers
- torchvision

### 依赖库说明

请在项目根目录下执行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

---

## 二、项目目录结构说明

```
project_root/
│
├── data/
│   ├── raw/                         # 原始数据（文本 + 图像）
│   └── processed/
│       ├── train/
│       │   ├── texts_cleaned/       # 训练集文本
│       │   └── images_processed/    # 训练集图像张量 (.pt)
│       ├── val/
│       │   ├── texts_cleaned/       # 验证集文本
│       │   └── images_processed/    # 验证集图像
│       ├── test/
│       │   ├── texts_cleaned/       # 测试集文本
│       │   └── images_processed/    # 测试集图像
│       ├── train_split.csv          # 训练集划分
│       ├── val_split.csv            # 验证集划分
│
├── datasets/
│   └── multimodal_attention_dataset.py
│
├── models/
│   └── multimodal_attention_model.py
│   └── text_encoder.py              # 文本特征化
│   └── image_encoder.py             # 图像特征化
│
├── scripts/
│   ├── preprocess_split.py            # 数据预处理脚本
|   ├── preprocess_text.py             # 文本清洗
|   ├── preprocess_image.py            # 图片处理
|   ├── preprocess_test.py             # 测试集预处理脚本
│   ├── train_multimodal_attention.py  # 多模态注意力融合模型
|   ├── train_multimodal_ablation.py   # 消融实验脚本
│   └── predict_test.py                # 测试集预测
│
├── config/
|   ├── label_map.py                       # 标签映射表
│   └── baseline_config.py                 # 模型与训练超参数配置
│
├── utils/
│   ├── metrics.py                         # 评估指标计算
│   └── result_logger.py                   # 训练日志工具
|
├── outputs/
│   ├── checkpoints/                   # 模型权重保存
│   ├── metrics/
|       └── attention_fusion_metrics.json   # 多模态注意力融合模型实验结果
│   └── results/
│       ├── val_metrics.csv                 # 验证集消融实验结果
│       └── test_predictions.csv            # 测试集预测结果
│
├── requirements.txt
└── README.md
```

---

## 三、实验流程

### 1、数据预处理
下载并存放原始数据后，运行：

```bash
python scripts/preprocess_text.py # 清洗文本
python scripts/preprocess_image.py # 图片处理
python scripts/preprocess_split.py # 划分数据集，生成对应索引，物理分开文本和图像
```

该步骤将完成：
- 文本编码修复与清洗
- 图像统一尺寸与格式处理
- 训练集 / 验证集8:2划分（固定随机种子:42）
- 生成对应的csv文件索引

### 2、多模态注意力融合模型

```bash
python scripts/train_multimodal_attention.py
```

该脚本将完成：
- 多模态 Attention Fusion 模型训练
- 在验证集上评估 Accuracy 与 Macro-F1，由此调整超参数
- 将结果保存至：
```
outputs/metrics/attention_fusion_metrics.json
```

### 3、测试集预测
```bash
python scripts/preprocess_test.py # 预处理测试集
python scripts/predict_test.py 
```
预测结果将输出为：
```text
outputs/results/test_predictions.csv
```
格式如下：
```csv
guid,pred_label
8,positive
1576,positive
2320,positive
```

### 4、消融实验

```bash
python scripts/train_multimodal_ablation.py
```

消融实验设置了以下三种模型变体：
- **Multimodal（完整模型）**：同时输入文本与图像特征，并通过跨模态注意力机制进行融合  
- **Text-only Ablation**：仅使用文本模态特征，去除图像模态输入  
- **Image-only Ablation**：仅使用图像模态特征，去除文本模态输入  
最后将各消融模型的验证结果统一保存至：
```text
outputs/results/val_metrics.csv
```

## 四、模型设计说明

- 文本编码器：BERT-base-uncased
- 图像编码器：ResNet-18（去除全连接层）
- 融合方式：跨模态注意力（Cross-Attention）
- 分类头：多层感知机（MLP）

通过Cross-Attention机制，模型学习文本与图像特征之间的相互依赖关系，从而提升情感分类性能。

---

## 五、评估指标

- Accuracy
- Macro-F1 Score

其中Macro-F1能更好地反映模型在类别不平衡情况下的整体性能。

---

## 六、参考资料
### GitHub 仓库参考
https://github.com/imadhou/multimodal-sentiment-analysis

---

## 七、说明
本项目为课程实验用途，所有实验结果仅用于模型结构与方法效果的验证与分析。
