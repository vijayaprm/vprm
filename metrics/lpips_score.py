import numpy as np
import lpips
import torch
from PIL import Image
import os

# Corrected function to load images, resize, convert to numpy array, and then to torch tensor
def load_images_as_tensors(image_paths, target_size=(256, 256)):
    tensors = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)  # Resize image
        img_np = np.array(img)  # Convert PIL Image to numpy array
        img_tensor = lpips.im2tensor(img_np)  # Correctly convert numpy array to tensor
        tensors.append(img_tensor)
    return torch.cat(tensors, dim=0)

# Calculate LPIPS (no changes needed here)
def calculate_lpips(real_images_dir, generated_images_dir):
    # Initialize LPIPS model with the preferred network
    lpips_model = lpips.LPIPS(net='alex')  # Using AlexNet as an example

    # Get paths of real and generated images
    real_image_paths = [os.path.join(real_images_dir, img) for img in os.listdir(real_images_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    generated_image_paths = [os.path.join(generated_images_dir, img) for img in os.listdir(generated_images_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Load, resize, and preprocess images
    real_images = load_images_as_tensors(real_image_paths, target_size=(256, 256))
    generated_images = load_images_as_tensors(generated_image_paths, target_size=(256, 256))

    # Compute LPIPS
    lpips_distance = lpips_model.forward(real_images, generated_images)

    return lpips_distance.mean().item()

# Example usage
real_images_dir = 'real'
generated_images_dir = 'predicted'
lpips_score = calculate_lpips(real_images_dir, generated_images_dir)
print("LPIPS score:", lpips_score)
