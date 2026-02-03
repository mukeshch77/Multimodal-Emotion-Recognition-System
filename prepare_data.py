import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split

# --- CONFIG ---
TEXT_CSV_PATH = 'data/emotion_dataset_raw.csv'
IMAGE_DIR = 'data/images' # FER2013 images ka root folder
OUTPUT_CSV_PATH = 'data/dataset.csv'

# Text labels ko Image folder ke labels se map karo
LABEL_MAP = {
    'joy': 'Happy',
    'sadness': 'Sad',
    'anger': 'Angry',
    'fear': 'Fear',
    'surprise': 'Surprise',
    'neutral': 'Neutral',
    'disgust': 'Disgust'
}

def create_dataset_csv():
    print("1. Loading text data...")
    text_df = pd.read_csv(TEXT_CSV_PATH)
    text_df.columns = ['label_raw', 'text']

    # 2. Label Cleaning and Mapping
    text_df['label'] = text_df['label_raw'].map(LABEL_MAP)
    # Jo labels map nahi hue (jaise 'shame'), unhe drop karo
    text_df.dropna(subset=['label'], inplace=True)
    text_df = text_df.reset_index(drop=True)
    print(f"Text data cleaned. Total entries: {len(text_df)}")

    # 3. Image Paths Collect karna
    image_paths = {}
    for label in LABEL_MAP.values():
        label_dir = os.path.join(IMAGE_DIR, label)
        if not os.path.isdir(label_dir):
            print(f"ERROR: Image folder not found: {label_dir}")
            print("Please ensure you have downloaded FER2013 and organized images under data/images/{Emotion}")
            return
        
        # Relative path store karna zaroori hai (e.g., 'images/Happy/123.png')
        paths = [
            os.path.join('images', label, f) 
            for f in os.listdir(label_dir) 
            if f.endswith(('.png', '.jpg'))
        ]
        image_paths[label] = paths
        print(f"Found {len(paths)} images for label: {label}")

    # 4. Random Pairing
    final_data = []
    
    # Text data ko shuffle karo for random assignment
    shuffled_indices = list(text_df.index)
    random.shuffle(shuffled_indices)
    text_df = text_df.loc[shuffled_indices].reset_index(drop=True)
    
    print("\n3. Randomly pairing text and images...")
    
    for index, row in text_df.iterrows():
        label = row['label']
        text = row['text']
        
        # Same emotion category se random image path choose karo
        if image_paths[label]:
            # Image paths ko loop mein use karne ke liye cyclic banao (agar images kam hon toh)
            selected_path = random.choice(image_paths[label]) 
            
            final_data.append({
                'image_path': selected_path,
                'text': text,
                'label': label
            })

    final_df = pd.DataFrame(final_data)
    
    # 5. Final CSV Save karna
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSuccessfully created final dataset: {OUTPUT_CSV_PATH} with {len(final_df)} entries.")
    
# Run the function
if __name__ == "__main__":
    create_dataset_csv()