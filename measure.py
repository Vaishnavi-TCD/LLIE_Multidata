# # # """
# # # # > Script for measuring quantitative performances in terms of
# # # #    - Structural Similarity Metric (SSIM) 
# # # #    - Peak Signal to Noise Ratio (PSNR)
# # # # > Maintainer: https://github.com/xahidbuffon
# # # """
# # # ## python libs
# # # import numpy as np
# # # from PIL import Image
# # # from glob import glob
# # # from os.path import join
# # # from ntpath import basename
# # # ## local libs
# # # from imqual_utils import getSSIM, getPSNR,uiqi,entropy




# # # ## compares avg ssim and psnr 
# # # def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(224,224)):
# # #     """
# # #         - gtr_dir contain ground-truths
# # #         - gen_dir contain generated images 
# # #     """
# # #     print('{Resizing images to ', im_res, '}')
# # #     gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
# # #     gen_paths = sorted(glob(join(gen_dir, "*.*")))
# # #     ssims, psnrs, mses,uiqi_vs,corr_coef,entropys = [], [], [],[],[],[]
# # #     for gtr_path, gen_path in zip(gtr_paths, gen_paths):
# # #         # print(gen_path)
# # #         gtr_f = basename(gtr_path).split('.')[0]
# # #         gen_f = basename(gen_path).split('.')[0]
# # #         if (gtr_f==gen_f):
# # #             # assumes same filenames
# # #             r_im = Image.open(gtr_path).resize(im_res)

# # #             g_im = Image.open(gen_path).resize(im_res)
            
# # #             # get ssim on RGB channels
# # #             ssim = getSSIM(np.array(r_im), np.array(g_im))
# # #             ssims.append(ssim)
# # #             # get psnr on L channel (SOTA norm)
# # #             r_im = r_im.convert('YCbCr'); g_im = g_im.convert('YCbCr')
# # #             psnr = getPSNR(np.array(r_im), np.array(g_im))
# # #             psnrs.append(psnr) 

# # #             uiqi_v0,uiqi_v1=uiqi(r_im,g_im)
# # #             uiqi_vs.append(uiqi_v0)
# # #             corr_coef.append(uiqi_v1) 
# # #             entropys.append(entropy(g_im))  
# # #     return np.array(ssims), np.array(psnrs), np.array(uiqi_vs),np.array(corr_coef),np.array(entropys)


# # # # """
# # # # Get datasets from
# # # #  - http://irvlab.cs.umn.edu/resources/euvp-dataset
# # # #  - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
# # # # """
# # # gtr_dir = "F:/spl_raqib/testing_code/results/gtr/"
# # # # gtr_dir = "F:/spl_raqib/gt/"

# # # # ## generated im paths
# # # gen_dir = "F:/spl_raqib/testing_code/results/UIEB/"
# # # # #gen_dir = "eval_data/ufo_test/deep-sesr/"  s


# # # # ### compute SSIM and PSNR
# # # SSIM_measures, PSNR_measures,uiqi_measures,corr_coef_measures,entropys_measures = SSIMs_PSNRs(gtr_dir, gen_dir)
# # # print ("SSIM on {0} samples".format(len(SSIM_measures)))
# # # print ("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

# # # print ("PSNR on {0} samples".format(len(PSNR_measures)))
# # # print ("Mea n: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

# # # print ("UIQI on {0} samples".format(len(uiqi_measures)))
# # # print ("Mea n: {0} std: {1}".format(np.mean(uiqi_measures), np.std(uiqi_measures)))

# # # print ("Entropy_Value on {0} samples".format(len(entropys_measures)))
# # # print ("Mea n: {0} std: {1}".format(np.mean(entropys_measures), np.std(entropys_measures)))


# import numpy as np
# from PIL import Image
# from glob import glob
# from os.path import join
# from ntpath import basename
# from imqual_utils import getSSIM, getPSNR, uiqi, entropy

# def split_concatenated_image(concatenated_image):
#     """
#     Split the concatenated image into output and reference images.
#     Modify this function according to your specific image format and splitting requirements.
#     """
#     # Example splitting: Assuming concatenated_image is [output | reference]
#     image_width = concatenated_image.shape[1]
#     split_index = image_width // 2
#     output_image = concatenated_image[:, split_index*0:split_index*1]
#     reference_image = concatenated_image[:, split_index*1:split_index*2]
#     return output_image, reference_image

# def SSIMs_PSNRs(concatenated_images, im_res=(256, 256)):
#     """
#     Measure quantitative performances (SSIM and PSNR) of concatenated images.

#     Args:
#         concatenated_images (list): List of concatenated images.
#         im_res (tuple): Desired image resolution.

#     Returns:
#         ssims (ndarray): Array of SSIM scores.
#         psnrs (ndarray): Array of PSNR scores.
#         uiqi_vs (ndarray): Array of UIQI scores.
#         corr_coef (ndarray): Array of correlation coefficients.
#         entropys (ndarray): Array of entropy values.
#     """
#     ssims, psnrs, uiqi_vs, corr_coef, entropys = [], [], [], [], []
#     for concatenated_image in concatenated_images:
#         # Split concatenated image into output and reference
#         output_image, reference_image = split_concatenated_image(concatenated_image)

#         # Resize images
#         r_im = Image.fromarray(reference_image.astype(np.uint8)).resize(im_res)
#         g_im = Image.fromarray(output_image.astype(np.uint8)).resize(im_res)

#         # Get SSIM on RGB channels
#         ssim = getSSIM(np.array(r_im), np.array(g_im))
#         ssims.append(ssim)

#         # Get PSNR on L channel
#         r_im = r_im.convert('L')
#         g_im = g_im.convert('L')
#         psnr = getPSNR(np.array(r_im), np.array(g_im))
#         print(psnr)
#         psnrs.append(psnr)

#         # Calculate UIQI and correlation coefficient
#         uiqi_v0, uiqi_v1 = uiqi(r_im, g_im)
#         uiqi_vs.append(uiqi_v0)
#         corr_coef.append(uiqi_v1)

#         # Calculate entropy
#         entropys.append(entropy(g_im))

#     return np.array(ssims), np.array(psnrs), np.array(uiqi_vs), np.array(corr_coef), np.array(entropys)

# gtr_dir='D:/Raqib/for_euvp_final/images_test_with_eca/'
# # Concatenated image paths
# concatenated_image_paths = sorted(glob(join(gtr_dir, "*.*")))

# # Read concatenated images
# concatenated_images = [np.array(Image.open(path)) for path in concatenated_image_paths]

# # Compute SSIM, PSNR, UIQI, and entropy
# SSIM_measures, PSNR_measures, uiqi_measures, corr_coef_measures, entropys_measures = SSIMs_PSNRs(concatenated_images)

# # Print results
# print("SSIM on {0} samples".format(len(SSIM_measures)))
# print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

# print("PSNR on {0} samples".format(len(PSNR_measures)))
# print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

# print("UIQI on {0} samples".format(len(uiqi_measures)))
# print("Mean: {0} std: {1}".format(np.mean(uiqi_measures), np.std(uiqi_measures)))

# print("Entropy on {0} samples".format(len(entropys_measures)))
# print("Mean: {0} std: {1}".format(np.mean(entropys_measures), np.std(entropys_measures)))



# import numpy as np
# from PIL import Image
# from glob import glob
# from os.path import join
# from ntpath import basename
# from imqual_utils import getSSIM, getPSNR, uiqi, entropy

# def split_concatenated_image(concatenated_image):
#     """
#     Split the concatenated image into output and reference images.
#     Modify this function according to your specific image format and splitting requirements.
#     """
#     # Example splitting: Assuming concatenated_image is [output | reference]
#     image_width = concatenated_image.shape[1]
#     split_index = image_width // 2
#     output_image = concatenated_image[:, :split_index]
#     reference_image = concatenated_image[:, split_index:]
#     return output_image, reference_image

# def SSIMs_PSNRs(concatenated_images, im_res=(256, 256)):
#     """
#     Measure quantitative performances (SSIM and PSNR) of concatenated images.

#     Args:
#         concatenated_images (list): List of concatenated images.
#         im_res (tuple): Desired image resolution.

#     Returns:
#         ssims (ndarray): Array of SSIM scores.
#         psnrs (ndarray): Array of PSNR scores.
#         uiqi_vs (ndarray): Array of UIQI scores.
#         corr_coef (ndarray): Array of correlation coefficients.
#         entropys (ndarray): Array of entropy values.
#     """
#     ssims, psnrs, uiqi_vs, corr_coef, entropys = [], [], [], [], []
#     for concatenated_image in concatenated_images:
#         # Split concatenated image into output and reference
#         output_image, reference_image = split_concatenated_image(concatenated_image)

#         # Resize images
#         r_im = Image.fromarray(reference_image.astype(np.uint8)).resize(im_res)
#         g_im = Image.fromarray(output_image.astype(np.uint8)).resize(im_res)

#         # Get SSIM on RGB channels
#         ssim = getSSIM(np.array(r_im), np.array(g_im))
#         ssims.append(ssim)

#         # Get PSNR on L channel
#         r_im = r_im.convert('L')
#         g_im = g_im.convert('L')
#         psnr = getPSNR(np.array(r_im), np.array(g_im))
#         psnrs.append(psnr)

#         # Calculate UIQI and correlation coefficient
#         uiqi_v0, uiqi_v1 = uiqi(r_im, g_im)
#         uiqi_vs.append(uiqi_v0)
#         corr_coef.append(uiqi_v1)

#         # Calculate entropy
#         entropys.append(entropy(g_im))

#         # Visualize the concatenated image
#         concatenated_image_visual = np.hstack((output_image, reference_image))
#         concatenated_image_visual = Image.fromarray(concatenated_image_visual.astype(np.uint8))
#         concatenated_image_visual.show()

#         print("SSIM:", ssim)
#         print("PSNR:", psnr)
#         print("UIQI:", uiqi_v0)
#         print("Correlation Coefficient:", uiqi_v1)
#         print("Entropy:", entropy(g_im))
#         print("----")

#     return np.array(ssims), np.array(psnrs), np.array(uiqi_vs), np.array(corr_coef), np.array(entropys)

# # Example usage
# gtr_dir = 'D:/Raqib/for_euvp_final/images_test_with_eca/'
# concatenated_image_paths = sorted(glob(join(gtr_dir, "*.*")))
# concatenated_images = [np.array(Image.open(path)) for path in concatenated_image_paths]
# SSIM_measures, PSNR_measures, uiqi_measures, corr_coef_measures, entropys_measures = SSIMs_PSNRs(concatenated_images)
# import numpy as np
# from PIL import Image
# from glob import glob
# from os.path import join
# from ntpath import basename
# from imqual_utils import getSSIM, getPSNR, uiqi, entropy

# def split_concatenated_image(concatenated_image):
#     """
#     Split the concatenated image into output and reference images.
#     Modify this function according to your specific image format and splitting requirements.
#     """
#     # Example splitting: Assuming concatenated_image is [output | reference]
#     image_width = concatenated_image.shape[1]
#     split_index = image_width // 2
#     output_image = concatenated_image[:, split_index*0:split_index*1]
#     reference_image = concatenated_image[:, split_index*1:split_index*2]
#     return output_image, reference_image

# def SSIMs_PSNRs(concatenated_images, im_res=(256, 256)):
#     """
#     Measure quantitative performances (SSIM and PSNR) of concatenated images.

#     Args:
#         concatenated_images (list): List of concatenated images.
#         im_res (tuple): Desired image resolution.

#     Returns:
#         ssims (ndarray): Array of SSIM scores.
#         psnrs (ndarray): Array of PSNR scores.
#         uiqi_vs (ndarray): Array of UIQI scores.
#         corr_coef (ndarray): Array of correlation coefficients.
#         entropys (ndarray): Array of entropy values.
#     """
#     ssims, psnrs, uiqi_vs, corr_coef, entropys = [], [], [], [], []
#     for i, concatenated_image in enumerate(concatenated_images):  # Add 'i' using enumerate
#         # Split concatenated image into output and reference
#         output_image, reference_image = split_concatenated_image(concatenated_image)

#         # Resize images
#         r_im = Image.fromarray(reference_image.astype(np.uint8)).resize(im_res)
#         g_im = Image.fromarray(output_image.astype(np.uint8)).resize(im_res)

#         # Get SSIM on RGB channels
#         ssim = getSSIM(np.array(r_im), np.array(g_im))
#         ssims.append(ssim)

#         # Get PSNR on L channel
#         r_im = r_im.convert('L')
#         g_im = g_im.convert('L')
#         psnr = getPSNR(np.array(r_im), np.array(g_im))
#         print("Image Pair:", basename(concatenated_image_paths[i]).split('.')[0])
#         print("SSIM:", ssim)
#         print("PSNR:", psnr)

#         # Calculate UIQI and correlation coefficient
#         uiqi_v0, uiqi_v1 = uiqi(r_im, g_im)
#         uiqi_vs.append(uiqi_v0)
#         corr_coef.append(uiqi_v1)

#         # Calculate entropy
#         entropys.append(entropy(g_im))

#         print("UIQI:", uiqi_v0)
#         print("Correlation Coefficient:", uiqi_v1)
#         print("Entropy:", entropy(g_im))
#         print("----")

#     return np.array(ssims), np.array(psnrs), np.array(uiqi_vs), np.array(corr_coef), np.array(entropys)

# # Example usage
# gtr_dir = 'D:/Raqib/for_euvp_final/images_test_with_eca/'
# concatenated_image_paths = sorted(glob(join(gtr_dir, "*.*")))

# concatenated_images = [np.array(Image.open(path)) for path in concatenated_image_paths]
# # print(concatenated_images)
# SSIM_measures, PSNR_measures, uiqi_measures, corr_coef_measures, entropys_measures = SSIMs_PSNRs(concatenated_images)



import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
from imqual_utils import getSSIM, getPSNR, uiqi, entropy

def split_concatenated_image(concatenated_image):
    """
    Split the concatenated image into output and reference images.
    Modify this function according to your specific image format and splitting requirements.
    """
    # Example splitting: Assuming concatenated_image is [output | reference]
    image_width = concatenated_image.shape[1]
    split_index = image_width // 2
    output_image = concatenated_image[:, split_index*0:split_index*1]
    reference_image = concatenated_image[:, split_index*1:split_index*2]
    return output_image, reference_image

def SSIMs_PSNRs(concatenated_images, im_res=(256, 256)):
    """
    Measure quantitative performances (SSIM and PSNR) of concatenated images.

    Args:
        concatenated_images (list): List of concatenated images.
        im_res (tuple): Desired image resolution.

    Returns:
        ssims (ndarray): Array of SSIM scores.
        psnrs (ndarray): Array of PSNR scores.
        uiqi_vs (ndarray): Array of UIQI scores.
        corr_coef (ndarray): Array of correlation coefficients.
        entropys (ndarray): Array of entropy values.
    """
    ssims, psnrs, uiqi_vs, corr_coef, entropys = [], [], [], [], []
    for concatenated_image in concatenated_images:
        # Split concatenated image into output and reference
        output_image, reference_image = split_concatenated_image(concatenated_image)

        # Resize images
        r_im = Image.fromarray(reference_image.astype(np.uint8)).resize(im_res)
        g_im = Image.fromarray(output_image.astype(np.uint8)).resize(im_res)

        # Get SSIM on RGB channels
        ssim = getSSIM(np.array(r_im), np.array(g_im))
        ssims.append(ssim)

        # Get PSNR on L channel
        r_im = r_im.convert('L')
        g_im = g_im.convert('L')
        psnr = getPSNR(np.array(r_im), np.array(g_im))
        # print(psnr)
        psnrs.append(psnr)

        # Calculate UIQI and correlation coefficient
        uiqi_v0, uiqi_v1 = uiqi(r_im, g_im)
        uiqi_vs.append(uiqi_v0)
        corr_coef.append(uiqi_v1)

        # Calculate entropy
        entropys.append(entropy(g_im))

    return np.array(ssims), np.array(psnrs), np.array(uiqi_vs), np.array(corr_coef), np.array(entropys)

gtr_dir='D:/Raqib/spl_raqib/UFO_test'
# Concatenated image paths
concatenated_image_paths = sorted(glob(join(gtr_dir, "*.*")))

# Read concatenated images
concatenated_images = [np.array(Image.open(path)) for path in concatenated_image_paths]

# Compute SSIM, PSNR, UIQI, and entropy
SSIM_measures, PSNR_measures, uiqi_measures, corr_coef_measures, entropys_measures = SSIMs_PSNRs(concatenated_images)

# Print results
print("SSIM on {0} samples".format(len(SSIM_measures)))
print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

print("PSNR on {0} samples".format(len(PSNR_measures)))
print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

print("UIQI on {0} samples".format(len(uiqi_measures)))
print("Mean: {0} std: {1}".format(np.mean(uiqi_measures), np.std(uiqi_measures)))

print("Entropy on {0} samples".format(len(entropys_measures)))
print("Mean: {0} std: {1}".format(np.mean(entropys_measures), np.std(entropys_measures)))