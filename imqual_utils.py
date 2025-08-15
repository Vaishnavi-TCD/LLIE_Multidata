"""
# > Implementation of the classic paper by Zhou Wang et. al.: 
#     - Image quality assessment: from error visibility to structural similarity
#     - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1284395
# > Maintainer: https://github.com/xahidbuffon
"""
from __future__ import division
import numpy as np
import math
from scipy.ndimage import gaussian_filter



def getSSIM(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim==2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[...,ch].astype(np.float64), Y[...,ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)


def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # variables are initialized as suggested in the paper
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5   

    # means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev 
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx  = (uxx - ux * ux) * unbiased_norm
    vy  = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim/D 
    mssim = SSIM.mean()

    return mssim



def getPSNR(X, Y):
    #assume RGB image
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    if rmse == 0: return 100
    else: return 20*math.log10(255.0/rmse)



def uiqi(img1, img2):
    # # Convert images to grayscale
    # img1 = np.mean(img1)
    # img2 = np.mean(img2)

    # Calculate means and standard deviations
    mean1, mean2 = np.mean(img1), np.mean(img2)
    std1, std2 = np.std(img1), np.std(img2)

    # Calculate covariance and correlation coefficients
    cov = np.mean((img1 - mean1) * (img2 - mean2))
    corr_coef = cov / (std1 * std2)

    # Calculate structural similarity
    k1 = 0.01
    k2 = 0.03
    L = 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1, sigma2 = np.var(img1), np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    # Calculate UIQI
    uiqi = corr_coef * ssim

    return uiqi,corr_coef


def entropy(img):
    # Convert the image to grayscale
    # gray = img.convert('L')
    
    # Calculate the histogram of the image
    hist = np.array(img.histogram()) / float(img.size[0] * img.size[1])
    
    # Calculate the Shannon entropy
    entropy = -np.sum([p * np.log2(p+1e-7) for p in hist if p != 0])
    
    return entropy    

