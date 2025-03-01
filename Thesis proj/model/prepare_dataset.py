import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define expected classes
EXPECTED_CLASSES = [
    'healthy',
    'early_blight',
    'late_blight',
    'tomato_yellow_leaf_curl_virus'
]

def verify_dataset_structure(source_dir):
    """Verify that all required disease class folders exist"""
    if not os.path.exists(source_dir):
        print(f"Error: raw_dataset folder not found at: {source_dir}")
        return False
        
    missing_classes = []
    for class_name in EXPECTED_CLASSES:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
            
    if missing_classes:
        print("\nMissing disease class folders:")
        for class_name in missing_classes:
            print(f"- {class_name}")
        print("\nPlease create these folders and add appropriate images.")
        return False
        
    return True

def preprocess_image(image_path):
    """Preprocess single image before dataset split"""
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    img = cv2.resize(img, (96, 96))
    
    # Optional: Apply histogram equalization
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def organize_dataset(source_dir, output_dir, validation_split=0.2):
    """
    Organize dataset into train and validation sets
    """
    # Create main directories
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Get all disease classes
    disease_classes = [d for d in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, d))]

    total_files = 0
    processed_files = 0
    
    # Count total files first
    for disease in disease_classes:
        image_files = [f for f in os.listdir(os.path.join(source_dir, disease))
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        total_files += len(image_files)
    
    print(f"Found {total_files} images in {len(disease_classes)} classes")
    
    for disease in disease_classes:
        print(f"\nProcessing {disease}...")
        # Create disease directories in train and validation
        os.makedirs(os.path.join(train_dir, disease), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, disease), exist_ok=True)

        # Get all images for this disease
        image_files = [f for f in os.listdir(os.path.join(source_dir, disease))
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Handle small datasets
        if len(image_files) < 2:
            print(f"Warning: {disease} has too few images ({len(image_files)})")
            print("Copying same image to both train and validation sets")
            train_files = image_files
            valid_files = image_files
        else:
            # Split into train and validation
            train_files, valid_files = train_test_split(
                image_files, 
                test_size=validation_split,
                random_state=42
            )

        print(f"- Found {len(image_files)} images")
        print(f"- Training: {len(train_files)}, Validation: {len(valid_files)}")
        
        # Copy files to respective directories
        for f in train_files:
            # Preprocess before copying
            img_path = os.path.join(source_dir, disease, f)
            processed_img = preprocess_image(img_path)
            output_path = os.path.join(train_dir, disease, f)
            cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            processed_files += 1
            print(f"\rProgress: {processed_files}/{total_files}", end="")
            
        for f in valid_files:
            shutil.copy2(
                os.path.join(source_dir, disease, f),
                os.path.join(valid_dir, disease, f)
            )
            processed_files += 1
            print(f"\rProgress: {processed_files}/{total_files}", end="")
    
    print("\nDataset organization completed!")
    print(f"Total images processed: {processed_files}")

if __name__ == "__main__":
    # Use absolute paths to avoid confusion
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SOURCE_DIR = os.path.join(PROJECT_ROOT, "raw_dataset")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
    
    print("Verifying dataset structure...")
    if not verify_dataset_structure(SOURCE_DIR):
        print("\nRequired folder structure:")
        for class_name in EXPECTED_CLASSES:
            print(f"/raw_dataset/{class_name}/")
        exit(1)
        
    organize_dataset(SOURCE_DIR, OUTPUT_DIR)
    print(f"Dataset organized successfully!")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
