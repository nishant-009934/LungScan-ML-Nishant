import numpy as np
import cv2

def apply_windowing(image, window_center, window_width):
    """
    Applies windowing (clipping) to a CT scan image.
    This helps focus on specific tissues (like lungs).
    
    Args:
        image (np.array): The raw CT scan in Hounsfield Units (HU).
        window_center (int): The center HU value we care about.
        window_width (int): The range of HU values we care about.
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    # Clip any values outside our min/max range
    windowed_image = np.clip(image, img_min, img_max)
    return windowed_image

def normalize_image(image):
    """
    Normalizes image pixel values to be between 0.0 and 1.0.
    Neural Networks learn much better with small numbers.
    """
    # Subtract the minimum value, then divide by the range
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val - min_val == 0:
        return image # Avoid dividing by zero if image is completely flat
        
    normalized = (image - min_val) / (max_val - min_val)
    return normalized

def resize_image(image, target_size=(128, 128)):
    """
    Resizes the image to a standard dimension.
    This is necessary because CNNs expect all images to be the exact same size.
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized

def preprocess_pipeline(image, is_hu=False):
    """
    Runs the complete preprocessing pipeline on a single image.
    
    Args:
        image (np.array): The raw image
        is_hu (bool): True if the image is raw DICOM data in Hounsfield Units.
                      False if it's our dummy PNG image.
    """
    # Step 1: If it's real CT data, apply lung windowing
    # Lungs typically have a center of -600 and width of 1500
    if is_hu:
        processed = apply_windowing(image, window_center=-600, window_width=1500)
    else:
        processed = image.copy()
        
    # Step 2: Normalize values to 0-1
    processed = normalize_image(processed)
    
    # Step 3: Resize to our standard size (e.g., 128x128)
    processed = resize_image(processed, target_size=(128, 128))
    
    # Step 4: Ensure it's in a format PyTorch likes (float32)
    processed = processed.astype(np.float32)
    
    return processed

if __name__ == "__main__":
    # Let's test it on a dummy random array
    dummy_scan = np.random.normal(-500, 200, (512, 512)) # Fake 512x512 scan
    print(f"Original shape: {dummy_scan.shape}, Original min/max: {np.min(dummy_scan):.2f}/{np.max(dummy_scan):.2f}")
    
    processed_scan = preprocess_pipeline(dummy_scan, is_hu=True)
    print(f"Processed shape: {processed_scan.shape}, Processed min/max: {np.min(processed_scan):.2f}/{np.max(processed_scan):.2f}")
    print("Preprocessing code works beautifully!")
