import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def generate_and_upscale(model_path, output_dir, num_images, class_names):
    """
    Loads a trained generator, generates images, upscales them to 224x224,
    and saves them for CNN training.
    """
    print(f"Loading Generator from {model_path}...")
    # Load model with custom_objects if needed, though usually Functional models load fine
    generator = models.load_model(model_path, compile=False)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    latent_dim = 100 # Match your training config
    num_classes = len(class_names)
    target_size = (224, 224)

    print(f"Generating {num_images} images per class...")
    
    total_generated = 0
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        print(f"Processing class: {class_name} ({class_idx})")
        
        # Generate in batches
        batch_size = 32
        generated_count = 0
        
        while generated_count < num_images:
            current_batch = min(batch_size, num_images - generated_count)
            
            # Prepare inputs
            noise = tf.random.normal(shape=(current_batch, latent_dim))
            labels = tf.convert_to_tensor(np.full((current_batch, 1), class_idx), dtype=tf.int32)
            
            # Generate
            gen_imgs = generator([noise, labels], training=False)
            gen_imgs = gen_imgs.numpy()
            
            # Process and Save
            for i in range(current_batch):
                img = gen_imgs[i]
                
                # Denormalize [-1, 1] -> [0, 255]
                img = ((img + 1.0) * 127.5).astype(np.uint8)
                
                # Upscale to 224x224
                # INTER_CUBIC or INTER_LINEAR is good for upscaling
                img_upscaled = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
                
                # Save
                filename = f"syn_{class_name}_{total_generated}_{i}.png"
                save_path = os.path.join(class_dir, filename)
                
                # Convert RGB to BGR for OpenCV saving
                img_bgr = cv2.cvtColor(img_upscaled, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, img_bgr)
            
            generated_count += current_batch
            total_generated += current_batch
            
    print(f"✓ Completed! Total images generated: {total_generated}")
    print(f"Images are saved in: {output_dir}")
    print(f"Each image has been upscaled to {target_size}")

if __name__ == "__main__":
    # Settings - Update these if paths change
    # Using the final checkpoint is PERFECT because your training was stable and improving until the very end.
    MODEL_PATH = "/content/drive/MyDrive/GAN_Malware_Detection/gan_training/checkpoints/generator_final.h5" 
    OUTPUT_DIR = "/content/augmented_data" # Save to local content first for speed, then copy to Drive if needed
    
    # Needs to match the class names from training
    CLASS_NAMES = ['Androm', 'Elex', 'Expiro', 'HackKMS', 'Hlux', 'Sality'] 
    
    NUM_IMAGES_PER_CLASS = 100 # Adjust as needed
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please ensure training is complete and 'gan_training/checkpoints/generator_final.h5' exists.")
        exit()

    generate_and_upscale(MODEL_PATH, OUTPUT_DIR, NUM_IMAGES_PER_CLASS, CLASS_NAMES)
