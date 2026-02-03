from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
import torch
import os # <-- Yeh add kiya

class MultimodalDataset(Dataset):
    def __init__(self, csv_data, img_dir, tokenizer, label2idx, transform=None):
        self.data = csv_data
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        label = self.label2idx[row['label']]
        image_path_relative = row['image_path'] # relative path

        # ----- TEXT -----
        tokens = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=128,
                                return_tensors='pt')

        # ----- IMAGE -----
        # Full path banaya: 'data/images/Happy/0001.png'
        image_path_full = os.path.join(self.img_dir, image_path_relative)
        
        # Image ko full path se load kiya
        image = Image.open(image_path_full).convert("RGB") 
        if self.transform:
            image = self.transform(image)

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'image': image,
            'label': torch.tensor(label)
        }