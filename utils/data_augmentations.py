import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter

# Random cropping and resizing
def crop_and_resize(image, target_size):
    # Random crop with resizing
    bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image)[0:3],  # Only pass height, width, and channels (batch dimension removed)
        bounding_boxes=tf.constant([[[0.0, 0.0, 1.0, 1.0]]]),
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.08, 1.0),
        use_image_if_no_bounding_boxes=True
    )

    ## Finding the bounding box for crop with resizing
    bbox_begin, bbox_size, _ = bbox
    image = tf.slice(image, bbox_begin, bbox_size)
    image = tf.image.resize(image, target_size)
    return image


# custom color distortion (souece: pseudo code in original paper's appendix)
def color_distortion(image):
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return tf.image.random_hue(image, max_delta=0.2)


# a function for Gaussian blur
def gaussian_blur(image):
    def blur(image_np):
        sigma = np.random.uniform(0.1, 2.0)
        kernel_radius = max(1, int(0.1 * min(image_np.shape[:2])))  # Dynamic kernel radius
        return gaussian_filter(image_np, sigma=sigma, truncate=kernel_radius)

    # Use TensorFlow's numpy_function to integrate Gaussian blur
    blurred_image = tf.numpy_function(blur, [image], tf.float32)
    return blurred_image


# a function for cutout
def cutout(image):
    cutout_size = tf.random.uniform(shape=[], minval=16, maxval=32, dtype=tf.int32)
    offset_x = tf.random.uniform(shape=[], minval=0, maxval=image.shape[1] - cutout_size, dtype=tf.int32)
    offset_y = tf.random.uniform(shape=[], minval=0, maxval=image.shape[0] - cutout_size, dtype=tf.int32)
    cutout_image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, cutout_size, cutout_size)
    return cutout_image


# function for Sobel filtering
def sobel_filter(image):
    # source https://stackoverflow.com/questions/56740582/how-to-calculate-sobel-edge-detection-in-tensorflow
    # reduce channel size
    if image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)

    sobel_edges = tf.image.sobel_edges(tf.expand_dims(image, axis=0))
    grad_mag_components = sobel_edges**2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1)
    grad_mag_img = tf.sqrt(grad_mag_square)
    return tf.reshape(grad_mag_img,  grad_mag_img.shape[1:])


# a function for applying gaussian noise
def gaussian_noise(image, noise_mean=0.1, noise_std=0.1):
    noise = tf.random.normal(tf.shape(image), mean=noise_mean, stddev=noise_std)
    noisy_image = image + noise
    return noisy_image

# Rotation functiom
def rotate(image):
    rotated_image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return rotated_image


# Augmentation functiojn
def augment_image(image, target_size=(256, 256), target_augmentations = ['random_crop_and_resize', 'horizontal_flip', 'color_distortion', 'gaussian_blur']):

    if 'random_crop_and_resize' in target_augmentations:
        # Crop and resize
        image = crop_and_resize(image, target_size)
    
    if 'horizontal_flip' in target_augmentations:
        # Random horizontal flip (0.5 probability)
        image = tf.image.random_flip_left_right(image)

    if 'color_distortion' in target_augmentations:
        # Apply color distortion
        image = color_distortion(image)

    if 'gaussian_blur' in target_augmentations:
        # Apply Gaussian blur
        image = gaussian_blur(image)

    if 'cutout' in target_augmentations:
        # Apply cutout
        image = cutout(image)

    if 'sobel_filter' in target_augmentations:
        # Apply Sobel filter
        image = sobel_filter(image)

    if 'gaussian_noise' in target_augmentations:
        # Apply Gaussian noise
        image = gaussian_noise(image)

    if 'rotate' in target_augmentations:
        # Apply rotation
        image = rotate(image)

    return image
