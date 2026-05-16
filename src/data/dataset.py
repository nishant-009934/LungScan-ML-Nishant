import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

# Import our preprocessing function
from src.data.preprocessing import preprocess_pipeline

class LungDataset(Dataset):
    """
    A custom PyTorch Dataset. It tells PyTorch how to find and load our images.
    Now equipped with Professional Data Augmentation for training.
    """
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        # 0 = Benign, 1 = Malignant
        classes = {'benign': 0, 'malignant': 1}
        
        for class_name, label in classes.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label)

        # Advanced Data Augmentation pipeline (only applied during training)
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor() # Converts to tensor and scales [0,1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply standard preprocessing (Resize, HU windowing if real DICOM, etc.)
        img_processed = preprocess_pipeline(img, is_hu=False)
        
        # OpenCV works with numpy float64/32. Ensure it's uint8 for PIL
        if img_processed.dtype != np.uint8:
            img_processed = (img_processed * 255).astype(np.uint8)
            
        # Apply PyTorch Augmentations (returns a Tensor of shape 1,H,W)
        img_tensor = self.transform(img_processed)
        
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return img_tensor, label_tensor

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset')
    my_dataset = LungDataset(dataset_path, is_train=True)
    print(f"Total images found: {len(my_dataset)}")
    if len(my_dataset) > 0:
        dataloader = DataLoader(my_dataset, batch_size=4, shuffle=True)
        images, labels = next(iter(dataloader))
        print(f"Batch Image Shape: {images.shape}")
        print(f"Batch Labels: {labels.flatten().tolist()}")
