import os
import yaml
import shutil
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm

def create_yolo_structure(base_dir="data/yolo_dataset"):
    """Creates the standard YOLO directory structure."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)
    return base_dir

def convert_xml_to_yolo(xml_path):
    """Parses VOC XML and returns YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    yolo_lines = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        # We only care about 'fire' class. 
        # If dataset has 'smoke', we can map it to 0 or ignore.
        # Let's map 'fire' to 0.
        if 'fire' in name.lower():
            cls_id = 0
        else:
            continue # Skip non-fire objects for now
            
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        
        # Convert to YOLO (x_center, y_center, width, height) normalized
        bb = ((b[0] + b[1]) / 2.0 / w, (b[2] + b[3]) / 2.0 / h, 
              (b[1] - b[0]) / w, (b[3] - b[2]) / h)
        
        yolo_lines.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
        
    return yolo_lines

def prepare_data(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    raw_dir = config['data']['raw_data_dir']
    yolo_dir = "data/yolo_dataset"
    create_yolo_structure(yolo_dir)
    
    print("Preparing Unified YOLO Dataset...")
    
    # --- 1. Process Indoor Dataset (Already YOLO format) ---
    indoor_base = os.path.join(raw_dir, "indoor", "Indoor Fire Smoke")
    indoor_splits = {'train': 'train', 'valid': 'val', 'test': 'test'}
    
    print("\nProcessing Indoor Dataset...")
    for src_split, dest_split in indoor_splits.items():
        src_img_dir = os.path.join(indoor_base, src_split, "images")
        src_lbl_dir = os.path.join(indoor_base, src_split, "labels")
        
        if not os.path.exists(src_img_dir):
            continue
            
        images = glob(os.path.join(src_img_dir, "*"))
        for img_path in tqdm(images, desc=f"Indoor {src_split}"):
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            
            # Copy Image
            shutil.copy(img_path, os.path.join(yolo_dir, dest_split, 'images', filename))
            
            # Copy Label
            lbl_name = name + ".txt"
            src_lbl = os.path.join(src_lbl_dir, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(yolo_dir, dest_split, 'labels', lbl_name))

    # --- 2. Process Outdoor (FlameVision) Dataset (XML format) ---
    # Path: data/raw/forest/FlameVision .../FlameVision/Detection
    # We need to find the exact path dynamically or hardcode based on inspection
    forest_base = glob(os.path.join(raw_dir, "forest", "**", "Detection"), recursive=True)[0]
    
    forest_splits = {'train': 'train', 'valid': 'val', 'test': 'test'}
    
    print("\nProcessing Outdoor Dataset...")
    for src_split, dest_split in forest_splits.items():
        src_img_dir = os.path.join(forest_base, src_split, "images")
        src_lbl_dir = os.path.join(forest_base, src_split, "annotations")
        
        if not os.path.exists(src_img_dir):
            continue
            
        images = glob(os.path.join(src_img_dir, "*"))
        for img_path in tqdm(images, desc=f"Outdoor {src_split}"):
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            
            # Copy Image
            # Handle potential duplicate filenames by prefixing
            new_filename = f"outdoor_{filename}"
            shutil.copy(img_path, os.path.join(yolo_dir, dest_split, 'images', new_filename))
            
            # Convert and Save Label
            xml_file = os.path.join(src_lbl_dir, name + ".xml")
            if os.path.exists(xml_file):
                yolo_lines = convert_xml_to_yolo(xml_file)
                if yolo_lines:
                    lbl_filename = f"outdoor_{name}.txt"
                    with open(os.path.join(yolo_dir, dest_split, 'labels', lbl_filename), 'w') as f:
                        f.write('\n'.join(yolo_lines))

    # --- 3. Create data.yaml ---
    data_yaml = {
        'path': os.path.abspath(yolo_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['fire']
    }
    
    with open(os.path.join(yolo_dir, "data.yaml"), 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"\nDataset preparation complete. Config saved to {os.path.join(yolo_dir, 'data.yaml')}")

if __name__ == "__main__":
    prepare_data()
