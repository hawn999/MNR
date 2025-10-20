# preprocess_data.py
import os
import glob
import numpy as np
import cv2
from tqdm import tqdm


def preprocess_and_save(source_dir, target_dir, image_size=80):
    """
    Reads original .npz files, processes images, and saves them to a new directory.

    Args:
        source_dir (str): Directory containing the original .npz files.
        target_dir (str): Directory where the processed .npz files will be saved.
        image_size (int): The target size for the images (e.g., 80x80).
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Find all original .npz files
    file_paths = glob.glob(os.path.join(source_dir, "*.npz"))

    if not file_paths:
        print(f"Warning: No .npz files found in '{source_dir}'")
        return

    print(f"Found {len(file_paths)} files to process. Starting...")

    # Process each file
    for file_path in tqdm(file_paths, desc="Processing files"):
        # Load the original data
        data = np.load(file_path)

        # --- Start: This logic is copied directly from your _get_data method ---
        # 1. Reshape the raw image data
        image = data["image"].reshape(16, 360, 640, 3)

        # 2. Prepare the array for resized images
        # The final shape will be (16, 3, H, W) for PyTorch
        resize_image = np.zeros((16, 3, image_size, image_size), dtype=np.uint8)

        # 3. Loop, resize each frame, and transpose dimensions
        for i in range(16):
            resized = cv2.resize(
                image[i],
                (image_size, image_size),
                interpolation=cv2.INTER_NEAREST
            )
            # Transpose from (H, W, C) to (C, H, W)
            resize_image[i] = resized.transpose(2, 0, 1)
        # --- End: Copied logic ---

        # Prepare the new data to be saved
        # We save the processed image and copy over all other data (like 'target')
        new_data = {key: data[key] for key in data.files if key != 'image'}
        new_data['image'] = resize_image

        # Define the output path
        base_filename = os.path.basename(file_path)
        output_path = os.path.join(target_dir, base_filename)

        # Save the new .npz file (compressed for efficiency)
        np.savez_compressed(output_path, **new_data)

    print(f"✨ Pre-processing complete! Processed files are saved in '{target_dir}'.")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Set your directories here
    ORIGINAL_DATA_DIR = '/home/scxhc1/RVP'  # 👈 SET THIS
    PROCESSED_DATA_DIR = '../dataset/datasets/RVP'  # 👈 SET THIS
    IMAGE_SIZE = 80  # 👈 Set your desired image size

    preprocess_and_save(ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE)