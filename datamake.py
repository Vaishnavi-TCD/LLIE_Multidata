# import os
# import cv2
# from glob import glob

# def rename_and_save_images(input_folder, output_folder, prefix="image", start_idx=1):
#     # Ensure the output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Get list of all image files in the input folder
#     image_paths = glob(os.path.join(input_folder, '*'))
    
#     # Loop through each image, rename and save to output folder
#     for idx, image_path in enumerate(image_paths, start=start_idx):
#         # Read the image
#         image = cv2.imread(image_path)
        
#         # Check if the image was loaded successfully
#         if image is None:
#             print(f"Error loading image: {image_path}")
#             continue
        
#         # Construct new filename with prefix and index
#         new_filename = f"Milk_{idx:04d}.jpg"   # Change extension if needed
#         new_filepath = os.path.join(output_folder, new_filename)

#         # Save the image with the new name
#         cv2.imwrite(new_filepath, image)
#         print(f"Saved {new_filepath}")

# # Example usage
# input_folder = 'D:/Raqib/UWT/IJCNNExten/uw_data/Turbid_Dataset/TURBID 3D/Milk'
# output_folder = 'D:/Raqib/UWT/IJCNNExten/uw_data/Turbid_Dataset/TURBID 3D/turbid/a'
# rename_and_save_images(input_folder, output_folder, prefix="renamed_image")


# import os
# import cv2

# def duplicate_image(input_image_path, output_folder, num_copies=20, prefix="duplicate"):
#     # Ensure the output folder exists
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Read the image
#     image = cv2.imread(input_image_path)
    
#     # Check if the image was loaded successfully
#     if image is None:
#         print(f"Error: Unable to load image from {input_image_path}")
#         return

#     # Save the image multiple times with unique names
#     for i in range(1, 22):
#         # Generate the filename for each duplicate
#         filename = f"Milk_{i:04d}.jpg"   # Change the extension if needed
#         filepath = os.path.join(output_folder, filename)
        
#         # Save the duplicate image
#         cv2.imwrite(filepath, image)
#         print(f"Saved duplicate: {filepath}")

# # Example usage
# input_image_path = 'D:/Raqib/UWT/IJCNNExten/uw_data/Turbid_Dataset/TURBID 3D/ref.jpg'
# output_folder = 'D:/Raqib/UWT/IJCNNExten/uw_data/Turbid_Dataset/TURBID 3D/turbid/b'
# duplicate_image(input_image_path, output_folder)


import os
import cv2
import random
import numpy as np

def apply_geometric_augmentation(image):
    """Apply random geometric transformations with only vertical and horizontal flips."""
    # Random rotation between -30 and 30 degrees
    angle = 90
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Random vertical or horizontal flip only
    flip_code = random.choice([0, 1])  # 0: vertical, 1: horizontal
    flipped = cv2.flip(rotated, flip_code)

    # Random scaling between 0.9x and 1.1x
    scale = random.uniform(0.9, 1.1)
    scaled = cv2.resize(flipped, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Crop or pad to original size if needed
    if scaled.shape[:2] != (h, w):
        scaled = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)

    return scaled

def augment_images(input_folder, reference_folder, output_input_folder, output_reference_folder):
    """Apply geometric transformations to images in input and reference folders."""
    os.makedirs(output_input_folder, exist_ok=True)
    os.makedirs(output_reference_folder, exist_ok=True)

    input_images = sorted(os.listdir(input_folder))
    reference_images = sorted(os.listdir(reference_folder))

    # Ensure both folders have the same number of images
    assert len(input_images) == len(reference_images), "Mismatch in number of input and reference images."

    for i, (input_img_name, ref_img_name) in enumerate(zip(input_images, reference_images)):
        # Load images
        input_img_path = os.path.join(input_folder, input_img_name)
        ref_img_path = os.path.join(reference_folder, ref_img_name)

        input_image = cv2.imread(input_img_path)
        ref_image = cv2.imread(ref_img_path)

        if input_image is None or ref_image is None:
            print(f"Error loading images: {input_img_name} or {ref_img_name}")
            continue

        # Apply geometric augmentations
        aug_input_image = apply_geometric_augmentation(input_image)
        aug_ref_image = apply_geometric_augmentation(ref_image)
        
        # Save augmented images with original name and index in separate folders
        input_name = os.path.splitext(input_img_name)[0]
        ref_name = os.path.splitext(ref_img_name)[0]
        
        aug_input_path = os.path.join(output_input_folder, f"{input_name}_{i+1:02d}.jpg")
        aug_ref_path = os.path.join(output_reference_folder, f"{ref_name}_{i+1:02d}.jpg")
        
        cv2.imwrite(aug_input_path, aug_input_image)
        cv2.imwrite(aug_ref_path, aug_ref_image)

        print(f"Saved augmented images: {aug_input_path} and {aug_ref_path}")

# Example usage
input_folder = 'D:/Raqib/UWT/IJCNNExten/uw_data/Turbid_Dataset/TURBID 3D/trubid/a'
reference_folder = 'D:/Raqib/UWT/IJCNNExten/uw_data/Turbid_Dataset/TURBID 3D/trubid/b'
output_input_folder = 'D:/Raqib/UWT/IJCNNExten/uw_data/Turbid_Dataset/TURBID 3D/a'
output_reference_folder = 'D:/Raqib/UWT/IJCNNExten/uw_data/Turbid_Dataset/TURBID 3D/b'

augment_images(input_folder, reference_folder, output_input_folder, output_reference_folder)
