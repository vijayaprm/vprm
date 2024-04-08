import os
import numpy as np
from PIL import Image

def calculate_mse(real_images_dir, generated_images_dir):
    # Get paths of real and generated images
    real_image_paths = [os.path.join(real_images_dir, img) for img in os.listdir(real_images_dir)]
    generated_image_paths = [os.path.join(generated_images_dir, img) for img in os.listdir(generated_images_dir)]

    # Ensure the number of images in both directories is the same
    if len(real_image_paths) != len(generated_image_paths):
        raise ValueError("Number of real images doesn't match the number of generated images")

    # Initialize mean squared error
    mse_sum = 0.0

    # Loop through images and calculate MSE
    for real_img_path, generated_img_path in zip(real_image_paths, generated_image_paths):
        # Open images and resize them to a consistent size
        real_img = Image.open(real_img_path).resize((224, 224)).convert('L')
        generated_img = Image.open(generated_img_path).resize((224, 224)).convert('L')

        # Convert images to numpy arrays
        real_img_array = np.array(real_img)
        generated_img_array = np.array(generated_img)

        # Calculate MSE for this pair of images
        mse = np.mean((real_img_array - generated_img_array) ** 2)

        # Accumulate MSE
        mse_sum += mse

    # Calculate mean MSE
    mean_mse = mse_sum / len(real_image_paths)
    
    return mean_mse

# Example usage
real_images_dir = 'real'
generated_images_dir = 'predicted'
mse_score = calculate_mse(real_images_dir, generated_images_dir)
print("Mean Squared Error (MSE) score:", mse_score)
