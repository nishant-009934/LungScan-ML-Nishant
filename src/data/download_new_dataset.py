import os
import shutil
import zipfile
import subprocess
import sys

def download_and_merge():
    print("Setting up Kaggle credentials...")
    os.environ['KAGGLE_USERNAME'] = 'n9sh1nt'
    os.environ['KAGGLE_KEY'] = '83485a500d58e1afe7359dfeee593686'
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_name = "adityamahimkar/iqothnccd-lung-cancer-dataset"
    zip_path = os.path.join(base_dir, "iqothnccd-lung-cancer-dataset.zip")
    extract_dir = os.path.join(base_dir, "temp_iqothnccd")
    
    dataset_dir = os.path.join(base_dir, "dataset")
    benign_dir = os.path.join(dataset_dir, "benign")
    malignant_dir = os.path.join(dataset_dir, "malignant")
    
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)
    
    print(f"Downloading {dataset_name}...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", base_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset. Ensure Kaggle is installed and key is valid. Error: {e}")
        return
        
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    print("Merging images into project dataset...")
    count = 0
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            src = os.path.join(root, file)
            folder_name = os.path.basename(root).lower()
            
            # Map classes to binary (Benign/Normal vs Malignant)
            if 'normal' in folder_name or 'benign' in folder_name:
                dst = os.path.join(benign_dir, f"iqothnccd_{folder_name}_{file}")
                shutil.copy2(src, dst)
                count += 1
            elif 'malignant' in folder_name or 'cancer' in folder_name:
                dst = os.path.join(malignant_dir, f"iqothnccd_{folder_name}_{file}")
                shutil.copy2(src, dst)
                count += 1
                
    print(f"Successfully merged {count} new images into the dataset!")
    
    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)

if __name__ == "__main__":
    download_and_merge()
