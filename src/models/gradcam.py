import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    This class extracts the internal 'thoughts' of our CNN to create a heatmap.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # We attach 'hooks' to the model. 
        # A hook is like a wiretap that listens to the math happening inside the layer.
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # Saves the feature map (what the layer "sees")
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # Saves the gradients (how important the features are for the final decision)
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_image, class_idx=None):
        """
        Generates a heatmap for a specific image.
        """
        # 1. Pass the image forward through the model
        model_output = self.model(input_image)
        
        # 2. If we don't specify a class, just use whatever the model predicted
        if class_idx is None:
            # We use sigmoid because we have 1 output node
            prob = torch.sigmoid(model_output)
            class_idx = 1 if prob.item() > 0.5 else 0
            
        self.model.zero_grad()
        
        # 3. Force the model to calculate gradients backwards from its prediction
        model_output.backward(retain_graph=True)
        
        # 4. Math magic: Multiply the 'activations' (what it saw) by the 'gradients' (how important it was)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # 5. Create the heatmap!
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap) # Ignore negative values
        heatmap /= torch.max(heatmap) # Normalize to 0-1
        
        return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):
    """
    Takes the raw heatmap and overlays it onto the original CT scan.
    """
    # Resize the tiny heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to colors (Red = high focus, Blue = low focus)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Convert original grayscale image to RGB so we can mix it with colors
    if len(original_img.shape) == 2:
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_img_rgb = original_img
        
    # Mix the original image (60%) with the heatmap (40%)
    superimposed_img = cv2.addWeighted(original_img_rgb, 0.6, heatmap_colored, 0.4, 0)
    
    return superimposed_img
