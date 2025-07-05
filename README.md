# se-msrnet
This repository contains the PyTorch implementation of SE-MSRNet, a novel convolutional neural network (CNN) designed for high-accuracy remote sensing image classification, as presented in the paper "Enhancing Remote Sensing Image Classification with SE-MSRNet: A Multi-Scale Residual Network Integrating Squeeze-and-Excitation Attention

> Submitted to *The Visual Computer* (Springer Journal)

---

## 📌 Highlights

- Combines **multi-scale convolutions (3×3, 5×5, 7×7)** with **Squeeze-and-Excitation (SE) attention**.
- Integrates the custom block into **ResNet-50**, forming **SE-MSRNet**.
- Outperforms baseline models across multiple remote sensing benchmarks: **AID, NWPU-RESISC45, EuroSAT, WHU-RS19**.
- Fully modular and built using **PyTorch**.

---

## 🗂️ Project Structure
├── ms_resnet.py # SE-MSRNet architecture definition
├── train.py # Training script
├── utils/
│ └── data_loader.py # Data loading utility
└── README.md # This file


---

## 🧠 Model Architecture

The core idea is to enhance standard ResNet-50 by adding a **Multi-Scale Residual Block** after the final feature map. This block:
- Uses 3×3, 5×5, and 7×7 convolutions
- Concatenates the outputs
- Applies SE attention to emphasize important channels
- Applies dropout for regularization

---
## 📊 Dataset Access

All datasets used in this study are publicly available. For WHU-RS19:

📁 **WHU-RS19 Dataset**:  
🔗 [https://pan.baidu.com/s/1OknFoQbxl0VIR1tXMcGE8A?pwd=eyx7](https://pan.baidu.com/s/1OknFoQbxl0VIR1tXMcGE8A?pwd=eyx7)  
🔐 Password: `eyx7` (Retrieved: April 3, 2025)

Other datasets used:
- [EuroSAT](https://github.com/phelber/EuroSAT)
- [AID](https://captain-whu.github.io/AID/)
- [NWPU-RESISC45](https://github.com/RSIA-LIESMARS-WHU/NWPU-RESISC45)

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision numpy

data/
├── WHU-RS19/
│   ├── class_1/
│   ├── class_2/
│   └── ...

