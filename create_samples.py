import os
import shutil
import random
from glob import glob

def create_samples():
    # Define source and destination
    # We use the processed YOLO dataset because it has unified labels
    src_base = "data/yolo_dataset/train" 
    dest_base = "samples"
    
    if os.path.exists(dest_base):
        shutil.rmtree(dest_base)
    os.makedirs(dest_base)
    
    # Create subfolders
    os.makedirs(os.path.join(dest_base, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_base, "labels"), exist_ok=True)
    
    # Get all images
    all_images = glob(os.path.join(src_base, "images", "*"))
    
    # Separate Indoor vs Outdoor (based on 'outdoor_' prefix we added earlier)
    indoor_imgs = [img for img in all_images if "outdoor_" not in os.path.basename(img)]
    outdoor_imgs = [img for img in all_images if "outdoor_" in os.path.basename(img)]
    
    # Select 15 random from each
    selected_indoor = random.sample(indoor_imgs, min(15, len(indoor_imgs)))
    selected_outdoor = random.sample(outdoor_imgs, min(15, len(outdoor_imgs)))
    
    print(f"Selecting {len(selected_indoor)} Indoor and {len(selected_outdoor)} Outdoor samples...")
    
    for img_path in selected_indoor + selected_outdoor:
        filename = os.path.basename(img_path)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        
        src_label = os.path.join(src_base, "labels", label_filename)
        
        # Copy Image
        shutil.copy(img_path, os.path.join(dest_base, "images", filename))
        
        # Copy Label (if exists)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(dest_base, "labels", label_filename))
            
    print("Samples created in 'samples/' folder.")

if __name__ == "__main__":
    create_samples()
