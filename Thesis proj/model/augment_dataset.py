import cv2
import numpy as np
import os
import tensorflow as tf

def create_augmentation_layer():
    """Create augmentation using tf.keras.Sequential"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        # Brightness adjustment
        tf.keras.layers.RandomBrightness(0.2)
    ])

def augment_dataset(input_dir, output_dir, samples_per_image=5):
    """Augment images in the dataset"""
    augmentation_layer = create_augmentation_layer()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        print(f"\nProcessing {class_name}...")
        
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            # Load and preprocess image
            img_path = os.path.join(class_dir, img_name)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.expand_dims(img, 0)
            
            # Save original image
            base_name = os.path.splitext(img_name)[0]
            original_path = os.path.join(output_class_dir, f"{base_name}_original.jpg")
            tf.io.write_file(
                original_path,
                tf.cast(img[0] * 255, tf.uint8).numpy()
            )
            
            # Generate augmented images
            for i in range(samples_per_image):
                aug_img = augmentation_layer(img, training=True)
                aug_img = tf.cast(aug_img[0] * 255, tf.uint8).numpy()
                output_path = os.path.join(output_class_dir, 
                                         f"{base_name}_aug_{i+1}.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
            print(f"Generated {samples_per_image} augmented images for {img_name}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(PROJECT_ROOT, "raw_dataset")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "augmented_dataset")
    SAMPLES_PER_IMAGE = 5  # Number of augmented images to generate per original image
    
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Generating {SAMPLES_PER_IMAGE} augmented images per original image")
    
    augment_dataset(INPUT_DIR, OUTPUT_DIR, SAMPLES_PER_IMAGE)
    print("\nAugmentation completed!")
