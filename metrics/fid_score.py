import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.linalg import sqrtm

# Function to load and preprocess images
def load_images(image_paths, target_size=(299, 299)):
    images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        images.append(img)
    return np.vstack(images)

# Function to compute FID score
def calculate_fid_score(real_images_dir, generated_images_dir):
    # Load InceptionV3 model (pre-trained on ImageNet)
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Get paths of real and generated images
    real_image_paths = [os.path.join(real_images_dir, img) for img in os.listdir(real_images_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    generated_image_paths = [os.path.join(generated_images_dir, img) for img in os.listdir(generated_images_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Load and preprocess images
    real_images = load_images(real_image_paths)
    generated_images = load_images(generated_image_paths)

    # Compute activations of Inception model for real and generated images
    real_activations = inception.predict(real_images)
    generated_activations = inception.predict(generated_images)

    # Compute mean and covariance statistics
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)

    # Correctly reshape sigma1 and sigma2 if necessary
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Compute square root of product of covariances, ensuring numerical stability
    eps = 1e-6
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = covmean.real + eps * np.eye(*covmean.shape)

    # Calculate the FID score
    diff = mu1 - mu2
    diff_sq = np.dot(diff, diff)
    trace = np.trace(sigma1 + sigma2 - 2.0 * sqrtm(covmean))

    fid_score = diff_sq + trace
    return fid_score

# Example usage
real_images_dir = 'real'
generated_images_dir = 'predicted'
fid_score = calculate_fid_score(real_images_dir, generated_images_dir)
print("Average FID score:", fid_score)
