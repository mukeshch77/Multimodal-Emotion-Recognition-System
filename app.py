# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from models.bert_resnet_fusion import BERT_ResNet_Fusion
import io
import os

# ----------------- CONFIG -----------------
MODEL_PATH = "fusion_model.pth"        # place your trained weights here
CSV_PATH = "data/dataset.csv"          # same CSV used in training (to rebuild LabelEncoder)
IMG_DIR = "data"                       # matches your MultimodalDataset img_dir
IMG_SIZE = (224, 224)
MAX_TEXT_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Multimodal Emotion Predictor", layout="centered")
st.title("🎭 Unified Multimodal Framework for Human Emotion Understanding")
st.write("Upload image and/or enter text. Uses your trained fusion_model.pth to predict emotion.")

# ------------- helpers: load label encoder from CSV -------------
@st.cache_data
def load_label_encoder(csv_path):
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at {csv_path}. Please ensure dataset.csv exists.")
        return None, None
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        st.error("dataset.csv must have a 'label' column (the final mapped labels like 'Happy', 'Angry', ...).")
        return None, None
    le = LabelEncoder()
    le.fit(df['label'].values)   # same as in your train.py
    classes = list(le.classes_)  # order will match training
    idx2label = {i: classes[i] for i in range(len(classes))}
    return le, idx2label

le, idx2label = load_label_encoder(CSV_PATH)
if idx2label is None:
    st.stop()

# ------------- load tokenizer and model -------------
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

@st.cache_resource
def load_model(model_path, num_classes):
    model = BERT_ResNet_Fusion(num_classes=num_classes)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Place fusion_model.pth in repo root.")
        return None
    # load state dict safely to CPU/GPU
    state = torch.load(model_path, map_location=DEVICE)
    # state likely is state_dict
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # try common wrapper keys
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            # attempt flexible load (if saved with DataParallel prefix)
            new_state = {}
            for k, v in state.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_k] = v
            model.load_state_dict(new_state)
    model.to(DEVICE)
    model.eval()
    return model

tokenizer = load_tokenizer()
model = load_model(MODEL_PATH, num_classes=len(idx2label))
if model is None:
    st.stop()

# ------------- transforms (inference) -------------
img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return img_transform(pil_img).unsqueeze(0)  # 1 x 3 x H x W

def preprocess_text(text):
    enc = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="pt",
    )
    return enc['input_ids'], enc['attention_mask']

# ------------- predict function -------------
def predict(image_pil=None, text=""):
    # Prepare text tensors (default zeros if no text)
    input_ids = torch.zeros((1, MAX_TEXT_LEN), dtype=torch.long, device=DEVICE)
    attention_mask = torch.zeros((1, MAX_TEXT_LEN), dtype=torch.long, device=DEVICE)
    if text and text.strip():
        in_ids, attn = preprocess_text(text)
        input_ids = in_ids.to(DEVICE)
        attention_mask = attn.to(DEVICE)

    # Prepare image tensor (zeros if no image)
    if image_pil is not None:
        img_tensor = preprocess_image(image_pil).to(DEVICE)
    else:
        img_tensor = torch.zeros((1, 3, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, img_tensor)  # expected (1, num_classes)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()  # shape (num_classes,)
        pred_idx = int(np.argmax(probs))
        pred_label = idx2label.get(pred_idx, str(pred_idx))
        return pred_label, probs

# ------------- UI -------------
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            st.image(image, caption="Uploaded image", use_column_width=True)
        except Exception as e:
            st.error("Could not read the image. Make sure the file is a valid image.")
            image = None
    else:
        image = None

with col2:
    text_input = st.text_area("Enter text (optional)", height=160, placeholder="Type sentence(s) here...")

if st.button("Predict emotion"):
    if (image is None) and (not text_input.strip()):
        st.warning("Please upload an image or enter text (or both).")
    else:
        with st.spinner("Running inference..."):
            pred_label, probs = predict(image, text_input)
            top_idx = int(np.argmax(probs))
            st.success(f"Predicted emotion: **{pred_label}**  (Confidence: {probs[top_idx]*100:.2f}%)")

            # show all probabilities sorted
            labels = [idx2label[i] for i in range(len(probs))]
            df = pd.DataFrame({"label": labels, "probability": probs})
            df = df.sort_values("probability", ascending=False).reset_index(drop=True)
            st.subheader("All class probabilities")
            st.dataframe(df)

            # bar chart
            st.bar_chart(data=df.set_index("label")["probability"])
