import os
import shutil
import zipfile
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings

# Suppress sklearn undefined metric warnings
warnings.filterwarnings('ignore')

# Import our custom files
from src.data.dataset import LungDataset
from src.models.cnn_model import AdvancedLungNet

def download_and_extract_kaggle():
    """
    Downloads the dataset via Kaggle API.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    zip_path = os.path.join(base_dir, 'chest-ctscan-images.zip')
    
    if not os.path.exists(zip_path):
        print("Downloading Kaggle Dataset...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", "mohamedhanyyy/chest-ctscan-images", "-p", base_dir], check=True)
        except Exception as e:
            print("ERROR: Could not download from Kaggle. Place the zip file manually.")
            return False
            
    print("Organizing dataset...")
    extract_dir = os.path.join(base_dir, 'temp_kaggle')
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    benign_dir = os.path.join(dataset_dir, 'benign')
    malignant_dir = os.path.join(dataset_dir, 'malignant')
    
    if not os.path.exists(benign_dir) or len(os.listdir(benign_dir)) < 50:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        os.makedirs(benign_dir, exist_ok=True)
        os.makedirs(malignant_dir, exist_ok=True)
        
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                src = os.path.join(root, file)
                folder_name = os.path.basename(root).lower()
                if 'normal' in folder_name:
                    shutil.copy2(src, os.path.join(benign_dir, f"{os.path.basename(os.path.dirname(root))}_{folder_name}_{file}"))
                elif 'adenocarcinoma' in folder_name or 'large' in folder_name or 'squamous' in folder_name:
                    shutil.copy2(src, os.path.join(malignant_dir, f"{os.path.basename(os.path.dirname(root))}_{folder_name}_{file}"))
                    
        shutil.rmtree(extract_dir)
        print("Dataset successfully extracted and organized into benign/malignant folders!")
    else:
        print("Dataset already organized. Skipping extraction.")
        
    return True

def train_model(max_epochs=20, batch_size=16):
    """
    Trains the CNN model and evaluates metrics.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset')
    
    print("\nStarting model training...")
    
    dataset = LungDataset(dataset_path, is_train=True)
    val_dataset_full = LungDataset(dataset_path, is_train=False) # No augmentations for validation
    
    if len(dataset) < 100:
        print(f"Error: Not enough images found ({len(dataset)}).")
        return
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # We use manual seeds to ensure both datasets split exactly the same way
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    _, val_dataset = torch.utils.data.random_split(val_dataset_full, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine Class Weights
    num_benign = len(os.listdir(os.path.join(dataset_path, 'benign')))
    num_malignant = len(os.listdir(os.path.join(dataset_path, 'malignant')))
    weight = torch.tensor([num_benign / (num_malignant + 1e-5)])
    
    model = AdvancedLungNet()
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    save_path = os.path.join(base_dir, 'models_saved', 'lung_cnn_model.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_f1 = 0.0
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Evaluation Phase
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.numpy().flatten())
                all_preds.extend(preds.numpy().flatten())
                all_labels.extend(labels.numpy().flatten())
                
        # Calculate Metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.0
            
        print(f"Epoch {epoch+1:02d}/{max_epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc*100:.1f}% | F1: {f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Scheduler steps based on F1-Score
        scheduler.step(f1)
        
        # Dynamic Checkpointing (Save only the best model)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"  * New best model saved! (F1: {f1:.4f})")
            
    print("\n--- Final Evaluation ---")
    print(f"   Accuracy    : {acc*100:.2f}%")
    print(f"   Precision   : {prec*100:.2f}%")
    print(f"   Sensitivity : {rec*100:.2f}%  (Recall)")
    print(f"   F1-Score    : {best_f1:.4f}")
    print(f"   AUC-ROC     : {roc_auc:.4f}")
    print("=======================================================\n")
    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    # download_and_extract_kaggle() bypassed since we already downloaded the massive IQ-OTH/NCCD dataset.
    train_model()
