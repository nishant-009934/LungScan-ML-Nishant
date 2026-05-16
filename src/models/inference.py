import os
import cv2
import torch
import matplotlib.pyplot as plt

# Import our custom files
from src.data.preprocessing import preprocess_pipeline
from src.models.cnn_model import SimpleCNN
from src.models.gradcam import GradCAM, overlay_heatmap

def predict_image(image_path):
    """
    The main Inference Pipeline!
    Takes a single image path and returns the prediction and a Grad-CAM heatmap.
    """
    print(f"Analyzing image: {image_path}")
    
    # 1. Load the Saved Model
    model = SimpleCNN()
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models_saved', 'lung_cnn_model.pt')
    
    if not os.path.exists(model_path):
        print("Error: Trained model not found!")
        return
        
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode
    
    # 2. Setup Grad-CAM
    # We want to look at the last Convolutional layer (conv3) because it holds the most complex features
    grad_cam = GradCAM(model, model.conv3)
    
    # 3. Load and Preprocess the Image
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        print("Error: Image could not be loaded.")
        return
        
    processed_img = preprocess_pipeline(original_img, is_hu=False)
    
    # Convert to PyTorch Tensor (Batch_size=1, Channels=1, H=128, W=128)
    input_tensor = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0)
    input_tensor.requires_grad = True # Required for Grad-CAM
    
    # 4. Make Prediction
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    
    if prob > 0.5:
        prediction = "Malignant (Cancer)"
        confidence = prob * 100
    else:
        prediction = "Benign (Safe)"
        confidence = (1 - prob) * 100
        
    print(f"\n--- PREDICTION ---")
    print(f"Result: {prediction}")
    print(f"Confidence: {confidence:.2f}%\n")
    
    # 5. Generate Grad-CAM Heatmap
    heatmap = grad_cam.generate_heatmap(input_tensor)
    
    # Overlay it on the image
    result_image = overlay_heatmap(original_img, heatmap)
    
    # Save the output image
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'prediction_result.png')
    
    # Matplotlib uses RGB, OpenCV uses BGR. Convert for saving.
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, result_image_bgr)
    
    print(f"Grad-CAM Heatmap saved to: {save_path}")
    print("Open the image to see exactly what the AI was looking at!")

if __name__ == "__main__":
    # Let's test it on one of our dummy malignant images
    test_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', 'malignant', 'malignant_1.png')
    predict_image(test_image)
