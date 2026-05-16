import os
import numpy as np
import cv2

def create_dummy_dataset(base_path, num_samples=20):
    """
    Creates a dummy dataset of 'benign' and 'malignant' images.
    This allows us to test our code pipeline immediately.
    """
    print(f"Generating {num_samples} dummy CT scans in {base_path}...")
    
    # Create folders
    benign_dir = os.path.join(base_path, 'benign')
    malignant_dir = os.path.join(base_path, 'malignant')
    
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create a blank 128x128 image with some noise (resembling CT noise)
        img = np.random.normal(loc=128, scale=20, size=(128, 128)).astype(np.uint8)
        
        # Add a "lung" shape (dark gray circle in the middle)
        cv2.circle(img, (64, 64), 50, (50), -1)
        
        if i % 2 == 0:
            # BENIGN: Add a smooth, round, bright nodule
            cv2.circle(img, (64, 64), 8, (200), -1)
            filepath = os.path.join(benign_dir, f'benign_{i}.png')
        else:
            # MALIGNANT: Add a spiky, irregular nodule
            # We do this by drawing a circle and then some lines coming out of it
            cv2.circle(img, (64, 64), 10, (200), -1)
            for angle in range(0, 360, 45):
                end_x = int(64 + 15 * np.cos(np.radians(angle)))
                end_y = int(64 + 15 * np.sin(np.radians(angle)))
                cv2.line(img, (64, 64), (end_x, end_y), (200), 2)
            filepath = os.path.join(malignant_dir, f'malignant_{i}.png')
            
        cv2.imwrite(filepath, img)

    print("Dummy dataset generated successfully!")

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset')
    create_dummy_dataset(dataset_path)
