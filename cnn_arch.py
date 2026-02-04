# ============================================
# CELL 5: Define CNN Model Architecture
# ============================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("="*60)
print("DEFINING CNN MODEL ARCHITECTURE")
print("="*60)

# Dataset parameters
NUM_CLASSES = 6
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

print(f"\nDataset Info:")
print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  Input shape: {INPUT_SHAPE}")

# Option 1: ResNet50 with Transfer Learning (RECOMMENDED for small dataset)
def create_resnet_model():
    """
    ResNet50 with transfer learning - best for limited data
    Uses ImageNet pre-trained weights
    """
    # Load pre-trained ResNet50 (without top classification layer)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE
    )

    # Freeze early layers (transfer learning)
    for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
        layer.trainable = False

    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

# Option 2: Custom CNN (Lighter, faster training)
def create_custom_cnn():
    """
    Custom CNN - lighter and faster for quick experiments
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        # Classification head
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

# Create model (choose one)
print("\nCreating ResNet50 model with transfer learning...")
baseline_model = create_resnet_model()

# Alternative: Use custom CNN for faster training
# baseline_model = create_custom_cnn()

# Compile model
baseline_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
baseline_model.summary()

# Count parameters
trainable_params = sum([tf.size(w).numpy() for w in baseline_model.trainable_weights])
total_params = sum([tf.size(w).numpy() for w in baseline_model.weights])

print("\n" + "="*60)
print("MODEL PARAMETERS")
print("="*60)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {total_params - trainable_params:,}")

# Setup callbacks for training
checkpoint_path = f'{PROJECT_DIR}/baseline_cnn_best.h5'
callbacks = [
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n✓ Model architecture defined!")
print(f"✓ Checkpoints will be saved to: {checkpoint_path}")
print("\nReady for baseline training!")
