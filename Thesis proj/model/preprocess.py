import cv2
import numpy as np
import os

def check_dataset(input_dir):
    """Check if dataset exists and contains images"""
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist!")
        print("Please create the following folder structure:")
        print(f"{input_dir}/")
        print("├── healthy/")
        print("├── early_blight/")
        print("├── late_blight/")
        print("└── tomato_yellow_leaf_curl_virus/")
        return False
        
    image_count = 0
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_count += len(images)
            print(f"Found {len(images)} images in {class_name}/")
    
    if image_count == 0:
        print("\nNo images found! Please add images to the appropriate folders:")
        print("1. Put healthy tomato images in 'healthy/'")
        print("2. Put early blight images in 'early_blight/'")
        print("3. Put late blight images in 'late_blight/'")
        print("4. Put mosaic virus images in 'tomato_yellow_leaf_curl_virus/'")
        return False
        
    print(f"\nTotal images found: {image_count}")
    return True

def preprocess_image(image_path, target_size=(224, 224)):
    """Enhanced preprocessing for tomato disease images"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    return img

def process_dataset(input_dir, output_dir):
    """Process entire dataset"""
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Create output class directory
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Process each image
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            input_path = os.path.join(class_path, img_name)
            output_path = os.path.join(output_class_path, 
                                     os.path.splitext(img_name)[0] + '.npy')
            
            # Preprocess and save as numpy array
            processed_img = preprocess_image(input_path)
            if processed_img is not None:
                np.save(output_path, processed_img)

if __name__ == "__main__":
    # Use absolute paths instead of relative paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(PROJECT_ROOT, "raw_dataset")
    processed_dir = os.path.join(PROJECT_ROOT, "processed_dataset")
    
    print(f"Looking for dataset in: {raw_dir}")
    print("Checking dataset structure...")
    if not check_dataset(raw_dir):
        exit(1)
    
    print("\nStarting preprocessing...")
    os.makedirs(processed_dir, exist_ok=True)
    process_dataset(raw_dir, processed_dir)
