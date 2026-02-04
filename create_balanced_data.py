import load_data

# ============================================
# CELL 4: Create Balanced Subset (5-6 Families)
# ============================================
import numpy as np

print("="*60)
print("CREATING SUBSET FROM FULL DATASET")
print("="*60 + "\n")

# Select specific malware families
# Choose families with 500 samples (good balance and visual distinctiveness)
SELECTED_FAMILIES = [
    'Androm',      # 500 samples
    'Elex',        # 500 samples
    'Expiro',      # 500 samples
    'HackKMS',     # 499 samples
    'Hlux',        # 500 samples
    'Sality'       # 499 samples
]

# You can also try these alternatives:
# 'Fasong', 'Injector', 'InstallCore', 'Neoreklami', 'Snarasite', 'Stantinko', 'VBA'

print("Selected families:")
selected_train_indices = []
selected_val_indices = []
selected_class_mapping = {}
subset_class_names = []

new_class_idx = 0
for family_name in SELECTED_FAMILIES:
    if family_name in class_names:
        old_idx = class_names.index(family_name)
        selected_class_mapping[old_idx] = new_class_idx

        # Get train indices
        train_mask = (y_train_full == old_idx)
        train_indices = np.where(train_mask)[0]
        selected_train_indices.extend(train_indices)

        # Get val indices
        val_mask = (y_val_full == old_idx)
        val_indices = np.where(val_mask)[0]
        selected_val_indices.extend(val_indices)

        subset_class_names.append(family_name)

        train_count = len(train_indices)
        val_count = len(val_indices)
        total_count = train_count + val_count

        print(f"  {new_class_idx}: {family_name:15s} - Train: {train_count:3d}, Val: {val_count:3d}, Total: {total_count:3d}")
        new_class_idx += 1
    else:
        print(f"  WARNING: {family_name} not found in dataset!")

# Create subset arrays
selected_train_indices = np.array(selected_train_indices)
selected_val_indices = np.array(selected_val_indices)

X_train = X_train_full[selected_train_indices]
y_train_old = y_train_full[selected_train_indices]

X_val = X_val_full[selected_val_indices]
y_val_old = y_val_full[selected_val_indices]

# Remap labels to 0, 1, 2, 3, 4, 5
y_train = np.array([selected_class_mapping[old_label] for old_label in y_train_old])
y_val = np.array([selected_class_mapping[old_label] for old_label in y_val_old])

print("\n" + "="*60)
print("SUBSET CREATED")
print("="*60)
print(f"Training set:   {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Total subset:   {len(X_train) + len(X_val)} images")
print(f"Number of classes: {len(subset_class_names)}")
print(f"Train memory: ~{X_train.nbytes / (1024**3):.2f} GB")
print(f"Val memory:   ~{X_val.nbytes / (1024**3):.2f} GB")

# Show subset distribution
print("\n" + "="*60)
print("SUBSET CLASS DISTRIBUTION")
print("="*60)
print(f"{'Class':<15s} {'Train':>8s} {'Val':>8s} {'Total':>8s}")
print("-"*60)

for idx in range(len(subset_class_names)):
    train_count = np.sum(y_train == idx)
    val_count = np.sum(y_val == idx)
    total_count = train_count + val_count
    print(f"{subset_class_names[idx]:<15s} {train_count:>8d} {val_count:>8d} {total_count:>8d}")

# Visualize subset samples
print("\nGenerating subset visualization...")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i in range(len(subset_class_names)):
    class_indices = np.where(y_train == i)[0]
    sample_idx = class_indices[0]

    axes[i].imshow(X_train[sample_idx])
    train_count = np.sum(y_train == i)
    val_count = np.sum(y_val == i)
    axes[i].set_title(f"{subset_class_names[i]}\nTrain: {train_count}, Val: {val_count}", fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(f'{PROJECT_DIR}/subset_samples.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Subset visualization saved to: {PROJECT_DIR}/subset_samples.png")

# Clean up full dataset to save memory
del X_train_full, y_train_full, X_val_full, y_val_full
import gc
gc.collect()

print("\n✓ Freed memory from full dataset")
print("✓ Subset ready for CNN training!")
