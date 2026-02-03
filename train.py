import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight # <-- Yeh import kiya
import numpy as np # <-- Yeh import kiya
from tqdm import tqdm
from models.bert_resnet_fusion import BERT_ResNet_Fusion # Assume this file is ready
from utils.dataloader import MultimodalDataset # Assume this file is updated

# --- CONFIG ---
csv_path = 'data/dataset.csv'
img_dir = 'data/'
batch_size = 8
epochs = 5
lr = 2e-5

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA ---
data = pd.read_csv(csv_path)

# Label Encoding
le = LabelEncoder()
data['label_int'] = le.fit_transform(data['label']) 
label2idx = {l: i for i, l in enumerate(le.classes_)}
num_classes = len(le.classes_)

# --- ACCURACY BOOST: 1. Data Augmentation and Normalization ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Augmentations add kiye for better generalization
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),     
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    # Normalization (ResNet standard)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

dataset = MultimodalDataset(data, img_dir, tokenizer, label2idx, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- ACCURACY BOOST: 2. Weighted Loss for Imbalanced Data ---
# Class weights calculate kiye
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(data['label']), 
    y=data['label'].values
)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# --- MODEL ---
model = BERT_ResNet_Fusion(num_classes=num_classes).to(device)

# Criterion mein weights pass kiye
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# --- TRAINING LOOP ---
print(f"Starting training on {device} with {num_classes} classes...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate accuracy for monitoring
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = total_loss / len(loader)
    epoch_accuracy = correct_predictions / total_samples * 100
    
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

torch.save(model.state_dict(), "fusion_model.pth")
print("\n✅ Training completed and model saved to fusion_model.pth.")