import os
import shutil
import zipfile

def setup_real_dataset():
    """
    This script organizes the downloaded Kaggle dataset.
    It combines the 3 cancer types into 'malignant' and 'normal' into 'benign'.
    """
    print("Setting up Real Kaggle Dataset...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # Check if the downloaded zip exists
    zip_path = os.path.join(base_dir, 'chest-ctscan-images.zip')
    
    if not os.path.exists(zip_path):
        print("Error: chest-ctscan-images.zip not found!")
        print("Please download it from Kaggle and place it in your Project folder.")
        return

    # Extract the zip
    extract_dir = os.path.join(base_dir, 'temp_kaggle_data')
    print("Extracting zip file (this might take a minute)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Prepare final folders
    benign_dir = os.path.join(dataset_dir, 'benign')
    malignant_dir = os.path.join(dataset_dir, 'malignant')
    
    # Delete dummy data if it exists
    if os.path.exists(benign_dir):
        shutil.rmtree(benign_dir)
    if os.path.exists(malignant_dir):
        shutil.rmtree(malignant_dir)
        
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    # Kaggle dataset usually has 'Data/train', 'Data/test', 'Data/valid'
    # We will combine all of them into our main folders for simplicity
    
    data_root = os.path.join(extract_dir, 'Data') # Depending on the zip structure, this might change
    
    # If the structure is directly train/test/valid
    if not os.path.exists(data_root):
        data_root = extract_dir
        
    subdirs = ['train', 'test', 'valid']
    
    images_moved = 0
    
    for split in subdirs:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            continue
            
        for folder in os.listdir(split_path):
            folder_path = os.path.join(split_path, folder)
            if not os.path.isdir(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                src_file = os.path.join(folder_path, filename)
                
                # Make filename unique
                new_filename = f"{split}_{folder}_{filename}"
                
                if folder.lower() == 'normal':
                    dst_file = os.path.join(benign_dir, new_filename)
                else:
                    # Adenocarcinoma, squamous, large cell all go to malignant
                    dst_file = os.path.join(malignant_dir, new_filename)
                    
                shutil.copy2(src_file, dst_file)
                images_moved += 1

    # Cleanup temp folder
    shutil.rmtree(extract_dir)
    print(f"Success! Organized {images_moved} real CT scan images into dataset folder.")
    print("You are now ready to re-run train.py!")

if __name__ == "__main__":
    setup_real_dataset()
