import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def load_malevis_presplit(base_path, img_size=(224, 224)):
    """
    Core logic for loading MaleVis dataset.
    """
    base_path = Path(base_path)
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    train_dir = base_path / 'train'
    val_dir = base_path / 'val'

    if not train_dir.exists():
        possible_dirs = list(base_path.rglob("train"))
        if possible_dirs:
            train_dir = possible_dirs[0]
            val_dir = train_dir.parent / 'val'

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    # Load training data
    for idx, class_dir in enumerate(class_dirs):
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        for img_path in tqdm(image_files, desc=f"  Loading {class_dir.name}", leave=False):
            try:
                img = cv2.imread(str(img_path))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X_train_list.append(img)
                y_train_list.append(idx)
            except Exception: continue

    # Load validation data
    val_class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    for idx, class_dir in enumerate(val_class_dirs):
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        for img_path in tqdm(image_files, desc=f"  Loading {class_dir.name}", leave=False):
            try:
                img = cv2.imread(str(img_path))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X_val_list.append(img)
                y_val_list.append(idx)
            except Exception: continue

    return np.array(X_train_list), np.array(y_train_list), \
           np.array(X_val_list), np.array(y_val_list), class_names

def get_full_dataset():
    """
    Convenience wrapper to call with default paths.
    """
    DATA_DIR = "/content/malevis_data/malevis_train_val_300x300"
    IMG_SIZE = (224, 224)
    return load_malevis_presplit(DATA_DIR, img_size=IMG_SIZE)

# This block ONLY runs if you do: python -m load_data
if __name__ == "__main__":
    print("="*60)
    print("EXECUTING LOAD_DATA DIRECTLY")
    print("="*60)
    
    X_train_full, y_train_full, X_val_full, y_val_full, class_names = get_full_dataset()
    
    print(f"Training set: {X_train_full.shape}")
    print(f"Validation set: {X_val_full.shape}")