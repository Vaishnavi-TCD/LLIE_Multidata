# """
# # > Script for measuring quantitative performances in terms of
# #    - Structural Similarity Metric (SSIM) 
# #    - Peak Signal to Noise Ratio (PSNR)
# # > Maintainer: https://github.com/xahidbuffon
# """
# ## python libs
# import numpy as np
# from PIL import Image
# from glob import glob
# from os.path import join
# from ntpath import basename
# ## local libs
# from imqual_utils import getSSIM, getPSNR,uiqi,entropy




# ## compares avg ssim and psnr 
# def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256,256)):
#     """
#         - gtr_dir contain ground-truths
#         - gen_dir contain generated images 
#     """
#     print('{Resizing images to ', im_res, '}')
#     gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
#     gen_paths = sorted(glob(join(gen_dir, "*.*")))
#     ssims, psnrs, mses,uiqi_vs,corr_coef,entropys = [], [], [],[],[],[]
#     for gtr_path, gen_path in zip(gtr_paths, gen_paths):
#         # print(gen_path)
#         gtr_f = basename(gtr_path).split('.')[0]
#         gen_f = basename(gen_path).split('.')[0]
#         if (gtr_f==gen_f):
#             # assumes same filenames
#             r_im = Image.open(gtr_path).resize(im_res)

#             g_im = Image.open(gen_path).resize(im_res)
           
#             # get ssim on RGB channels
#             ssim = getSSIM(np.array(r_im), np.array(g_im))
#             ssims.append(ssim)
#             # get psnt on L channel (SOTA norm)
#             r_im = r_im.convert("L"); g_im = g_im.convert("L")
#             psnr = getPSNR(np.array(r_im), np.array(g_im))
#             psnrs.append(psnr) 

#             uiqi_v0,uiqi_v1=uiqi(r_im,g_im)
#             uiqi_vs.append(uiqi_v0)
#             corr_coef.append(uiqi_v1) 
#             entropys.append(entropy(g_im))  
#     return np.array(ssims), np.array(psnrs), np.array(uiqi_vs),np.array(corr_coef),np.array(entropys)


# # """
# # Get datasets from
# #  - http://irvlab.cs.umn.edu/resources/euvp-dataset
# #  - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
# # """
# gtr_dir = "F:/spl_raqib/gt/"
# # #gtr_dir = "/home/xahid/datasets/released/UFO-120/TEST/hr/"

# # ## generated im paths
# gen_dir = "F:/spl_raqib/out/"
# # #gen_dir = "eval_data/ufo_test/deep-sesr/"  


# # ### compute SSIM and PSNR
# SSIM_measures, PSNR_measures,uiqi_measures,corr_coef_measures,entropys_measures = SSIMs_PSNRs(gtr_dir, gen_dir)
# print ("SSIM on {0} samples".format(len(SSIM_measures)))
# print ("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

# print ("PSNR on {0} samples".format(len(PSNR_measures)))
# print ("Mea n: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

# print ("UIQI on {0} samples".format(len(uiqi_measures)))
# print ("Mea n: {0} std: {1}".format(np.mean(uiqi_measures), np.std(uiqi_measures)))

# print ("Entropy_Value on {0} samples".format(len(entropys_measures)))
# print ("Mea n: {0} std: {1}".format(np.mean(entropys_measures), np.std(entropys_measures)))


import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
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





# Optional: define expected image shape (useful if input/output/reference sizes are known)
def split_image(img):
    h, w, c = img.shape
    part_w = w // 3
    input_img = img[:, :part_w]
    output_img = img[:, part_w:2*part_w]
    reference_img = img[:, 2*part_w:]
    return input_img, output_img, reference_img
psnr_list = []
ssim_list = []
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Failed to read image: {filename}")
            continue
        input_img, output_img, ref_img = split_image(img)
        # Convert to grayscale if needed for SSIM
        gray_output = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        # Compute metrics
        psnr_val = psnr(ref_img, output_img, data_range=255)
        ssim_val = ssim(gray_ref, gray_output, data_range=255)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        print(f"{filename} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
# Averages
print("\nAverage PSNR:", np.mean(psnr_list), np.size(psnr_list))
print("Average SSIM:", np.mean(ssim_list))