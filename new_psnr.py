import os

import cv2

import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim
 
# 🔍 Converts BGR image to Y channel only

def rgb_to_y_channel(img):

    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    y_channel = ycbcr[:, :, 0]  # Extract Y component

    return y_channel
 
# 📐 Splits concatenated image into input, output, and reference parts

def split_image(img):

    h, w, c = img.shape

    part_w = w // 3

    input_img = img[:, :part_w]

    output_img = img[:, part_w:2*part_w]

    reference_img = img[:, 2*part_w:]

    return input_img, output_img, reference_img
 


# Path to the folder containing concatenated images
# folder_path = "./images_test" # -- mixed model test results
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/images_test/real"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/images_test/syn"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/images_test/v1"

#best model test results
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_mixed"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv2_real"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv2_syn"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv1"

# cross models testing -- Mixed
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_mixed_lolv1"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_mixed_lolv2_real"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_mixed_lolv2_syn"

# # cross models testing -- LolV1
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv1_lolv1"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv1_lolv2_real"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv1_lolv2_syn"

# # cross models testing -- LolV2_Real
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv2_real"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv2_real_lolv1"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv2_real_lolv2_syn"

# # cross models testing -- LolV2_Syn
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv2_syn_lolv1"
# folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv2_syn_lolv2_real"
folder_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_test_outputs/best_lolv2_syn_lolv2_syn"

 
psnr_list = []

ssim_list = []
 
# 🌀 Main loop over images

for filename in os.listdir(folder_path):

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

        filepath = os.path.join(folder_path, filename)

        img = cv2.imread(filepath)
 
        if img is None:

            print(f"❌ Failed to read image: {filename}")

            continue
 
        input_img, output_img, ref_img = split_image(img)
 
        # 🎯 Convert to Y channel for PSNR

        y_output = rgb_to_y_channel(output_img)

        y_ref = rgb_to_y_channel(ref_img)
 
        # ⚙️ Compute metrics

        psnr_val = psnr(y_ref, y_output, data_range=255)
 
        # 🖤 SSIM on grayscale

        gray_output = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        ssim_val = ssim(gray_ref, gray_output, data_range=255)
 
        psnr_list.append(psnr_val)

        ssim_list.append(ssim_val)
 
        print(f"{filename} - PSNR (Y): {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
 
# 📊 Print Averages

print("\n📈 Average PSNR (Y): {:.2f} dB over {} images".format(np.mean(psnr_list), len(psnr_list)))

print("📈 Average SSIM: {:.4f}".format(np.mean(ssim_list)))

 