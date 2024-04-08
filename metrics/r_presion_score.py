import os

def calculate_r_precision(real_images_dir, generated_images_dir):
    # Get paths of real and generated images
    real_image_paths = [os.path.join(real_images_dir, img) for img in os.listdir(real_images_dir)]
    generated_image_paths = [os.path.join(generated_images_dir, img) for img in os.listdir(generated_images_dir)]

    # Initialize relevant count
    relevant_count = 0

    # Loop through real images and check if they exist in generated images
    for real_img_path in real_image_paths:
        if real_img_path in generated_image_paths:
            relevant_count += 1

    # Calculate R-Precision
    r_precision = relevant_count / len(real_image_paths)
    
    return r_precision

# Example usage
real_images_dir = 'real'
generated_images_dir = 'predicted'
r_precision_score = calculate_r_precision(real_images_dir, generated_images_dir)
print("R-Precision score:", r_precision_score)
