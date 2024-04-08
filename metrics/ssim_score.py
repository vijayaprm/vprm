from skimage.metrics import structural_similarity as ssim
from skimage import io
from skimage.transform import resize
import os

def calculate_similarity(real_images_dir, generated_images_dir, target_size=(256, 256), win_size=7):
    # Get paths of real and generated images
    real_image_paths = [os.path.join(real_images_dir, img) for img in os.listdir(real_images_dir)]
    generated_image_paths = [os.path.join(generated_images_dir, img) for img in os.listdir(generated_images_dir)]

    # Ensure the number of images in both directories is the same
    if len(real_image_paths) != len(generated_image_paths):
        raise ValueError("Number of real images doesn't match the number of generated images")

    # Initialize total similarity
    total_similarity = 0.0

    # Loop through images and calculate SSIM
    for real_img_path, generated_img_path in zip(real_image_paths, generated_image_paths):
        # Read and resize images
        real_img = io.imread(real_img_path)
        generated_img = io.imread(generated_img_path)
        real_img_resized = resize(real_img, target_size)
        generated_img_resized = resize(generated_img, target_size)

        # Calculate SSIM
        similarity = ssim(real_img_resized, generated_img_resized, multichannel=True, win_size=win_size)

        # Accumulate similarity
        total_similarity += similarity

    # Calculate average similarity
    average_similarity = total_similarity / len(real_image_paths)

    return average_similarity

# Example usage
real_images_dir = 'real'
generated_images_dir = 'predicted'
similarity_score = calculate_similarity(real_images_dir, generated_images_dir)
print("Average Structural Similarity Index (SSIM) score:", similarity_score)
