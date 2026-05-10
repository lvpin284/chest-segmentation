# 超声胸腔积液智能分析 Agent
# Ultrasound Pleural Effusion Intelligent Analysis Agent

---

## 项目简介 / Overview

本项目实现了一个基于深度学习的超声胸腔积液智能分析 Agent，具备以下能力：

This project implements a deep-learning-powered intelligent agent for analysing pleural effusion in ultrasound images, with the following capabilities:

| 功能 | Description |
|------|-------------|
| 🔬 图像分割 | U-Net 语义分割模型，自动识别积液区域 |
| 📊 定量分析 | 计算积液比例、估算面积、判断严重程度 |
| 🤖 智能报告 | 基于 LLM 生成双语（中/英）临床描述 |
| 🖼️ 可视化 | 生成叠加分割掩码的可视化报告图 |

---

## 项目结构 / Project Structure

```
chest-_segmentation/
├── src/
│   ├── models/
│   │   └── unet.py              # U-Net model + BCE+Dice loss
│   ├── preprocessing/
│   │   └── image_processor.py   # CLAHE, resize, normalise, augmentation
│   ├── agent/
│   │   └── pleural_effusion_agent.py  # Main agent (LangChain + U-Net)
│   └── utils/
│       └── visualization.py     # Overlay, metrics, figures
├── tests/
│   ├── test_model.py
│   ├── test_preprocessing.py
│   ├── test_visualization.py
│   └── test_agent.py
├── examples/
│   └── demo.py                  # CLI demo script
├── requirements.txt
├── setup.py
└── pyproject.toml
```

---

## 快速上手 / Quick Start

### 1. 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 运行演示 / Run Demo

```bash
# 使用合成测试图像（无需真实超声图像）
python examples/demo.py --synthetic

# 分析真实超声图像
python examples/demo.py --image path/to/ultrasound.png

# 保存可视化叠加图
python examples/demo.py --synthetic --save-overlay output/overlay.png
```

### 3. 在代码中使用 / Programmatic Usage

```python
from src.agent.pleural_effusion_agent import PleuralEffusionAgent

# 初始化 Agent（可选: 提供模型权重路径和 OpenAI API Key）
agent = PleuralEffusionAgent(
    weights_path="path/to/unet_weights.pth",  # 可选
    # llm=None  → 使用规则引擎; 设置 OPENAI_API_KEY 环境变量启用 LLM
)

# 分析超声图像
report = agent.analyze("ultrasound_scan.png")

# 打印报告
print(report)

# 获取结构化数据
data = report.to_dict()
print(data["severity"])
print(data["effusion_ratio"])
print(data["findings"])
```

---

## 模块说明 / Module Details

### U-Net 分割模型 (`src/models/unet.py`)

经典 U-Net 架构（Ronneberger et al., 2015），针对医学图像分割优化：

- **输入**: 单通道灰度超声图像 (1 × H × W)  
- **输出**: 积液区域概率图 (1 × H × W)  
- **损失函数**: BCE + Dice 联合损失，有效处理类别不平衡

```python
from src.models.unet import UNet, SegmentationLoss

model = UNet(in_channels=1, num_classes=1, base_features=64)
loss_fn = SegmentationLoss(bce_weight=0.5)
```

### 图像预处理 (`src/preprocessing/image_processor.py`)

超声图像专用预处理流程：

1. **CLAHE** 对比度自适应直方图均衡化
2. **保持宽高比的 resize + zero-pad**
3. **归一化** 至 `[0, 1]`
4. **转换为 PyTorch Tensor**

```python
from src.preprocessing.image_processor import UltrasoundPreprocessor

preprocessor = UltrasoundPreprocessor(image_size=(256, 256))
tensor = preprocessor("scan.png")  # shape: (1, 256, 256)
```

### 训练增强 (`src/preprocessing/image_processor.py`)

```python
from src.preprocessing.image_processor import TrainingAugmentor

augmentor = TrainingAugmentor(
    image_size=(256, 256),
    flip_prob=0.5,
    rotation_degrees=10.0,
    noise_std=0.02,
)
tensor = augmentor("scan.png")
```

### 可视化工具 (`src/utils/visualization.py`)

```python
from src.utils.visualization import overlay_mask, severity_from_ratio

# 叠加分割结果
overlay = overlay_mask(image, mask, alpha=0.4)

# 积液严重程度分级
severity = severity_from_ratio(effusion_ratio)
# 返回: "少量积液 (Small effusion)" 等
```

---

## LLM 集成 / LLM Integration

Agent 支持任何兼容 LangChain `BaseChatModel` 的 LLM：

```bash
# 使用 OpenAI GPT-4o（需要 API Key）
export OPENAI_API_KEY="sk-..."
python examples/demo.py --synthetic
```

不配置 API Key 时，Agent 将自动使用内置的**规则引擎**生成报告，无需联网。

---

## 积液严重程度分级 / Severity Classification

| 积液比例 | 严重程度 |
|----------|---------|
| < 2%     | 无明显积液 (No significant effusion) |
| 2% – 8%  | 少量积液 (Small effusion) |
| 8% – 20% | 中量积液 (Moderate effusion) |
| ≥ 20%    | 大量积液 (Large effusion) |

---

## 训练指南 / Training Guide

若您有标注数据集，可按以下方式训练模型：

```python
import torch
from torch.utils.data import DataLoader
from src.models.unet import UNet, SegmentationLoss
from src.preprocessing.image_processor import TrainingAugmentor

model = UNet(in_channels=1, num_classes=1).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = SegmentationLoss(bce_weight=0.5)

# ... 准备 DataLoader，然后标准训练循环 ...
for epoch in range(100):
    for images, masks in dataloader:
        optimizer.zero_grad()
        logits = model(images.cuda())
        loss = loss_fn(logits, masks.cuda())
        loss.backward()
        optimizer.step()

# 保存权重
torch.save(model.state_dict(), "unet_weights.pth")
```

---

## 运行测试 / Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 环境变量 / Environment Variables

| 变量名 | 说明 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥，用于 LLM 报告生成 |
| `UNET_WEIGHTS_PATH` | U-Net 模型权重文件路径 |

---

## 参考文献 / References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
- Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797). 3DV.
