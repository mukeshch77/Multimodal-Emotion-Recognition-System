# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from transformers import BertTokenizer
from models.bert_resnet_fusion import BERT_ResNet_Fusion
import io

# ----------------- CONFIG -----------------
MODEL_REPO = "mukeshch77/multimodal-emotion-model"
MODEL_FILE = "fusion_model.pth"

IMG_SIZE = (224, 224)
MAX_TEXT_LEN = 128
DEVICE = torch.device("cpu")  # streamlit cloud safe

# labels must be SAME ORDER as training
IDX2LABEL = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

st.set_page_config(page_title="Multimodal Emotion Predictor", layout="centered")
st.title("🎭 Unified Multimodal Framework for Human Emotion Understanding")
st.write("Upload image and/or enter text to predict emotion.")

# ----------------- LOAD TOKENIZER -----------------
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

# ----------------- LOAD MODEL FROM HF -----------------
@st.cache_resource
def load_model(num_classes):
    model = BERT_ResNet_Fusion(num_classes=num_classes)

    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
    state = torch.hub.load_state_dict_from_url(
        url,
        map_location="cpu",
        progress=True
    )

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

tokenizer = load_tokenizer()
model = load_model(num_classes=len(IDX2LABEL))

# ----------------- IMAGE TRANSFORMS -----------------
img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return img_transform(pil_img).unsqueeze(0)

def preprocess_text(text):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]

# ----------------- PREDICT -----------------
def predict(image_pil=None, text=""):
    input_ids = torch.zeros((1, MAX_TEXT_LEN), dtype=torch.long)
    attention_mask = torch.zeros((1, MAX_TEXT_LEN), dtype=torch.long)

    if text.strip():
        ids, mask = preprocess_text(text)
        input_ids = ids
        attention_mask = mask

    if image_pil is not None:
        img_tensor = preprocess_image(image_pil)
    else:
        img_tensor = torch.zeros((1, 3, IMG_SIZE[0], IMG_SIZE[1]))

    with torch.no_grad():
        logits = model(input_ids, attention_mask, img_tensor)
        probs = F.softmax(logits, dim=1).numpy().squeeze()
        pred_idx = int(np.argmax(probs))
        return IDX2LABEL[pred_idx], probs

# ----------------- UI -----------------
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload image (optional)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.image(image, caption="Uploaded image", use_column_width=True)
    else:
        image = None

with col2:
    text_input = st.text_area(
        "Enter text (optional)",
        height=160,
        placeholder="Type sentence(s) here..."
    )

if st.button("Predict emotion"):
    if image is None and not text_input.strip():
        st.warning("Please upload an image or enter text.")
    else:
        with st.spinner("Running inference..."):
            pred_label, probs = predict(image, text_input)
            top_idx = int(np.argmax(probs))

            st.success(
                f"Predicted emotion: **{pred_label}** "
                f"(Confidence: {probs[top_idx]*100:.2f}%)"
            )

            labels = [IDX2LABEL[i] for i in range(len(probs))]
            df = pd.DataFrame({
                "label": labels,
                "probability": probs
            }).sort_values("probability", ascending=False)

            st.subheader("All class probabilities")
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("label"))
