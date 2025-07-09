# 🚗 Parking Space Detection using Vision Transformers

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0%2B-yellow)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

Parking space detection is a key component of intelligent transportation systems aimed at alleviating urban congestion and improving parking efficiency. This project was developed as a **group capstone project** for our Computer Vision coursework which investigates the application of **Vision Transformers (ViTs)** in comparison with traditional Convolutional Neural Networks (CNNs) for parking occupancy detection. Our goal was to develop a **few-shot learning model** capable of accurate segmentation using minimal labeled data, making it scalable to new parking environments without extensive annotation. We utilized the ACPDS (fully annotated) and PKLot (partially annotated) datasets and applied preprocessing to convert classification-style data into pixel-level segmentation maps. A pretrained DINOv2 ViT model was used as the feature extractor, followed by experimentation with both linear probing and a SegFormer-style decoder head for segmentation. The results show that the ViT-based architecture significantly outperforms CNN baselines, achieving high F1 scores even under challenging conditions such as occlusions, varying weather, and camera distortions. This work highlights the robustness, scalability, and practicality of Vision Transformers for real-world parking space detection systems.

## 📝 How It Works

1. **DINOv2** extracts rich visual features from parking lot images using self-supervised learning
2. **Custom SegFormer decoder** processes these features for pixel-level classification
3. **Few-shot learning** allows the model to adapt to new parking environments with minimal data
4. **Real-time processing** analyzes video streams and outputs segmentation masks
5. **Comprehensive evaluation** measures performance using F1, Precision, Recall, and IoU metrics

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: Pre-trained Vision Transformer models
- **DINOv2**: Self-supervised Vision Transformer backbone
- **OpenCV**: Computer vision and video processing
- **NumPy & Matplotlib**: Data processing and visualization
- **Torchmetrics**: Comprehensive evaluation metrics
- **Jupyter Notebooks**: Interactive development and experimentation

## 📦 Project Structure

```
├── src/
│   ├── model.py                       # Core model architectures
│   ├── dataset.py                     # Data loading and preprocessing
│   ├── model_test.py                  # Comprehensive evaluation framework
│   └── video_test.py                  # Real-time video processing
├── notebooks/
│   ├── dinoXformer.ipynb             # DINOv2 + Linear classifier
│   ├── segformer_acpds.ipynb         # SegFormer on ACPDS dataset
│   ├── segformer_pklot.ipynb         # SegFormer on PKLot dataset
│   └── simple_linear_probing.ipynb   # Linear probing baseline
├── weights/                          # Trained model checkpoints
├── test/                             # Test videos and outputs
├── CV_Project_Report.pdf             # Detailed technical report
└── README.md                         # Project documentation
```

## 🖥️ Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- 16GB+ RAM for dataset processing

### Install Dependencies

```bash
pip install torch torchvision transformers datasets
pip install opencv-python matplotlib pillow numpy
pip install torchmetrics tqdm evaluate
```

### Dataset Setup

**Option A: ACPDS Dataset (Fully Annotated)**

```bash
# Download ACPDS dataset and organize:
├── ACPDS/
│   ├── ACPDS/
│   │   ├── images/
│   │   └── int_masks/
```

**Option B: PKLot Dataset (Partially Annotated)**

```bash
# Download PKLot dataset and organize:
├── PKLOT/
│   ├── PKLOT/
│   │   ├── images/
│   │   └── int_masks/
```

### Train Models

**DINOv2 + Linear Probing:**

```bash
jupyter notebook dinoXformer.ipynb
```

**SegFormer Architecture:**

```bash
jupyter notebook segformer_acpds.ipynb  # For ACPDS dataset
jupyter notebook segformer_pklot.ipynb  # For PKLot dataset
```

**Baseline Comparison:**

```bash
jupyter notebook simple_linear_probing.ipynb
```

### Evaluate Performance

```bash
python model_test.py  # Comprehensive evaluation with metrics
```

### Real-Time Video Processing

```bash
python video_test.py  # Process parking lot videos
```

### Running Tests

To evaluate model performance:

```bash
python model_test.py
```

To process test videos:

```bash
python video_test.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
