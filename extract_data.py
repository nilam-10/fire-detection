import yaml
import os
import zipfile
from pathlib import Path

def extract_data():
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found.")
        return

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['data']['raw_data_dir']
    
    # Indoor
    indoor_zip = config['data']['indoor_zip_path']
    indoor_extract_to = os.path.join(raw_dir, "indoor")
    
    # Forest
    forest_zip = config['data']['forest_zip_path']
    forest_extract_to = os.path.join(raw_dir, "forest")
    
    # Extract
    for zip_path, extract_to in [(indoor_zip, indoor_extract_to), (forest_zip, forest_extract_to)]:
        # Handle relative paths
        if not os.path.isabs(zip_path):
            zip_path = os.path.abspath(zip_path)
            
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} to {extract_to}...")
            os.makedirs(extract_to, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_to)
                print("Extraction complete.")
            except Exception as e:
                print(f"Error extracting {zip_path}: {e}")
        else:
            print(f"Warning: Zip file not found: {zip_path}")
            print("Please ensure you have downloaded the dataset zip files and placed them in the 'data/' directory.")

if __name__ == "__main__":
    extract_data()
