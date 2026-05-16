import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

# Import our custom files
from src.data.dataset import LungDataset
from src.models.cnn_model import AdvancedLungNet

def evaluate_model():
    print("Starting Model Evaluation...")
    
    # 1. Setup Data
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset')
    
    # Ideally, this should be a separate 'test' folder, but we use the dummy data for now
    test_dataset = LungDataset(dataset_path, is_train=False) 
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 2. Load the trained model
    model = AdvancedLungNet()
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models_saved', 'lung_cnn_model.pt')
    
    if not os.path.exists(model_path):
        print("Model file not found! Please run train.py first.")
        return
        
    # Load the saved 'brain' into our model architecture
    model.load_state_dict(torch.load(model_path))
    model.eval() # Tell model we are testing, NOT training
    
    all_preds = []
    all_labels = []
    
    print("Running predictions...")
    with torch.no_grad(): # Don't calculate gradients (saves memory/time)
        for images, labels in test_loader:
            outputs = model(images)
            
            # The model outputs raw numbers. We use sigmoid to turn them into probabilities (0.0 to 1.0)
            probs = torch.sigmoid(outputs)
            
            # If probability > 50%, predict Malignant (1), else Benign (0)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            
    # 3. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n--- EVALUATION RESULTS ---")
    print(f"Accuracy:  {acc*100:.2f}% (Can be misleading!)")
    print(f"Precision: {precision*100:.2f}% (When it says cancer, how often is it right?)")
    print(f"Recall:    {recall*100:.2f}% (Out of all real cancers, how many did it find?)")
    print(f"F1-Score:  {f1*100:.2f}% (Balance of Precision & Recall)")
    
    print("\nConfusion Matrix:")
    print("                 Predicted Benign | Predicted Malignant")
    print(f"Actual Benign:          {cm[0][0]}         |        {cm[0][1]}")
    print(f"Actual Malignant:       {cm[1][0]}         |        {cm[1][1]}")
    
    if cm[1][0] > 0:
        print("\nWARNING: You have False Negatives! The model missed some cancers. This is dangerous.")

if __name__ == "__main__":
    evaluate_model()
