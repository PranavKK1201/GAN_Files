# ============================================
# CELL 3: Load Dataset into Memory (Pre-split Version)
# ============================================
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def load_malevis_presplit(base_path, img_size=(224, 224)):
    """
    Load MaleVis dataset that's already split into train/val folders.
    Structure: base_path/train/class_name/*.png
               base_path/val/class_name/*.png
    """
    base_path = Path(base_path)

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    class_names = []

    # Get class names from train folder
    train_dir = base_path / 'train'
    val_dir = base_path / 'val'

    if not train_dir.exists():
        # Try to find the actual train folder
        possible_dirs = list(base_path.rglob("train"))
        if possible_dirs:
            train_dir = possible_dirs[0]
            val_dir = train_dir.parent / 'val'

    print(f"Train directory: {train_dir}")
    print(f"Val directory: {val_dir}")

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    print(f"\nFound {len(class_names)} classes")
    print("="*60)

    # Load training data
    print("\nLOADING TRAINING DATA:")
    print("-"*60)
    for idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))

        print(f"Class {idx}: {class_name:20s} - {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"  Loading {class_name}", leave=False):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)

                X_train_list.append(img)
                y_train_list.append(idx)

            except Exception as e:
                continue


    # Load validation data
    print("\n" + "-"*60)
    print("LOADING VALIDATION DATA:")
    print("-"*60)
    val_class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])

    for idx, class_dir in enumerate(val_class_dirs):
        class_name = class_dir.name
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))

        print(f"Class {idx}: {class_name:20s} - {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"  Loading {class_name}", leave=False):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)

                X_val_list.append(img)
                y_val_list.append(idx)

            except Exception as e:
                continue

    # Convert to numpy arrays
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)

    return X_train, y_train, X_val, y_val, class_names

# Load the dataset
print("="*60)
print("LOADING MALEVIS DATASET (PRE-SPLIT)")
print("="*60 + "\n")

# Point to the correct directory
DATA_DIR = "/content/malevis_data/malevis_train_val_300x300"
IMG_SIZE = (224, 224)

X_train_full, y_train_full, X_val_full, y_val_full, class_names = load_malevis_presplit(
    DATA_DIR,
    img_size=IMG_SIZE
)

print("\n" + "="*60)
print("DATASET LOADED SUCCESSFULLY!")
print("="*60)
print(f"Training set:   {X_train_full.shape} images")
print(f"Validation set: {X_val_full.shape} images")
print(f"Total images:   {len(X_train_full) + len(X_val_full)}")
print(f"Number of classes: {len(class_names)}")
print(f"Train memory: ~{X_train_full.nbytes / (1024**3):.2f} GB")
print(f"Val memory:   ~{X_val_full.nbytes / (1024**3):.2f} GB")

# Display class names
print(f"\nClass names ({len(class_names)} total):")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Show distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
print(f"{'Class':<20s} {'Train':>8s} {'Val':>8s} {'Total':>8s}")
print("-"*60)

for idx in range(len(class_names)):
    train_count = np.sum(y_train_full == idx)
    val_count = np.sum(y_val_full == idx)
    total_count = train_count + val_count
    print(f"{class_names[idx]:<20s} {train_count:>8d} {val_count:>8d} {total_count:>8d}")

# Visualize samples
print("\nGenerating visualization...")
n_samples = min(10, len(class_names))
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i in range(n_samples):
    class_indices = np.where(y_train_full == i)[0]
    if len(class_indices) > 0:
        sample_idx = class_indices[0]
        axes[i].imshow(X_train_full[sample_idx])
        axes[i].set_title(f"{class_names[i]}", fontsize=9)
        axes[i].axis('off')

for i in range(n_samples, 10):
    axes[i].axis('off')

PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'

plt.tight_layout()
plt.savefig(f'{PROJECT_DIR}/full_dataset_samples.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Visualization saved to: {PROJECT_DIR}/full_dataset_samples.png")
print("\n✓ Dataset loaded with pre-existing train/val split!")
print("✓ Ready for subsetting!")
