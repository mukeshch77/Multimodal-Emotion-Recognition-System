# 🎭 Unified Multimodal Framework for Human Emotion Understanding

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-emotion-recognition-system-yxfnrwzmiv6wrvj6d5j7ig.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/mukeshch77)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A robust deep learning framework that performs **multimodal emotion recognition** by fusing visual cues (facial expressions) and textual context. This system integrates **BERT** for text analysis and **ResNet** for image processing into a unified fusion architecture to predict human emotions with high accuracy.

## 🚀 Live Demo
Experience the real-time inference app here:
**[👉 Launch Live App](https://multimodal-emotion-recognition-system-yxfnrwzmiv6wrvj6d5j7ig.streamlit.app/)**

---

## Approach & Model Architecture

This project solves the problem of emotion ambiguity by using **Late Fusion** of two powerful modalities. Single-modality models often fail when facial expression contradicts text (e.g., a sarcastic smile). Our unified framework captures both.

### The Architecture: `BERT_ResNet_Fusion`
The model consists of three main blocks:

1.  **Visual Encoder (ResNet):**
    * Uses a pre-trained **ResNet** (Residual Neural Network) backbone.
    * Extracts high-level spatial features from facial images (sized 224x224).
    * Outputs a feature vector representing the visual emotion.

2.  **Text Encoder (BERT):**
    * Uses **`bert-base-uncased`** from Hugging Face Transformers.
    * Processes input text (up to 128 tokens) to extract contextual semantic embeddings.
    * Captures nuance, sentiment, and intent from the text.

3.  **Fusion Layer & Classifier:**
    * Concatenates the visual and textual feature vectors.
    * Passes the combined vector through fully connected layers (Dense Layers).
    * **Output:** Softmax probability distribution over **7 Emotion Classes**.

### Supported Emotions
The system predicts the following 7 classes:
| 😡 Angry | 🤢 Disgust | 😨 Fear | 😄 Happy | 😐 Neutral | 😢 Sad | 😲 Surprise |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |

---

## 📂 Project Structure

```text
MULTIMODAL_EMOTION_RECOGNITION
├── .venv/                   # Virtual environment
├── data/                    # Dataset storage
│   ├── images/              # Raw images organized by class
│   └── dataset.csv          # Text-Image pairings
├── models/                  # Model definitions
│   └── bert_resnet_fusion.py # Core model architecture class
├── utils/                   # Helper scripts
│   └── dataloader.py        # Custom PyTorch Dataset class
├── app.py                   # Streamlit Inference Application
├── config.yaml              # Configuration parameters
├── fusion_model.pth         # Trained model weights
├── prepare_data.py          # Data preprocessing script
├── train.py                 # Training loop & validation
└── requirements.txt         # Project dependencies
```
## 🛠️ Installation & Local Setup
Follow these steps to run the project locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/mukeshch77/Multimodal-Emotion-Recognition-System.git](https://github.com/mukeshch77/Multimodal-Emotion-Recognition-System.git)
cd Multimodal-Emotion-Recognition-System
```
### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the App
```bash
streamlit run app.py
```

## How It Works (Inference Pipeline)
The app.py handles the end-to-end inference flow without requiring the training dataset.

### 1. Input: 
- User uploads an image and/or enters text.

### 2.Preprocessing:

- Image: Resized to (224, 224), normalized using ImageNet mean/std.

- Text: Tokenized using BertTokenizer (padding/truncation to 128 tokens).

### 3. Model Loading:

- Weights are fetched securely from Hugging Face Hub (mukeshch77/multimodal-emotion-model) cached locally.

- No huge .pth files need to be committed to GitHub.

### 4. Prediction: 
- The Fusion Model aggregates features and returns the confidence score for all 7 classes.
