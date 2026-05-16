import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import LungDataset
from src.models.cnn_model import AdvancedLungNet

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset')
    model_path = os.path.join(base_dir, 'models_saved', 'lung_cnn_model.pt')
    
    print("Loading Validation Dataset...")
    # Use same seed/split as training to get the exact validation set
    val_dataset_full = LungDataset(dataset_path, is_train=False)
    
    train_size = int(0.8 * len(val_dataset_full))
    val_size = len(val_dataset_full) - train_size
    
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = torch.utils.data.random_split(val_dataset_full, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print("Loading AdvancedLungNet weights...")
    model = AdvancedLungNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running Inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.numpy().flatten())
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            
    # Calculate exact metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0
        
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n=======================================================")
    print("   VERIFIED MODEL METRICS")
    print("=======================================================")
    print(f"Accuracy    : {acc*100:.2f}%")
    print(f"Precision   : {prec*100:.2f}%")
    print(f"Recall      : {rec*100:.2f}%")
    print(f"F1-Score    : {f1:.4f}")
    print(f"AUC-ROC     : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("=======================================================\n")

if __name__ == "__main__":
    evaluate()
