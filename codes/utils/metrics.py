import numpy as np
import math
import cv2
import torch

def to_y_channel(img):
    """
    Converts image to Y channel.
    Args:
        img (np.ndarray): Image in range [0, 255] (uint8) or [0, 1] (float). BGR or RGB order?
                          Assumes BGR (cv2 default).
    Returns:
        np.ndarray: Y channel in range [0, 1] (float).
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        # BGR to YCbCr (MATLAB standard coefficients)
        # Y = 65.481 * R + 128.553 * G + 24.966 * B + 16
        # normalized to [0, 1]:
        # Y = 0.257 * R + 0.504 * G + 0.098 * B + 16/255
        # Note: cv2.COLOR_BGR2YCrCb does this but keys are different order.
        # Let's use manual dot product for precision matching MATLAB.
        
        # BGR order:
        B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        Y = 65.481 * R + 128.553 * G + 24.966 * B + 16.0
        Y /= 255.0
    else:
        Y = img
    return Y

def calculate_psnr(img1, img2, crop_border=0):
    """
    Calculate PSNR on Y channel.
    Args:
        img1 (np.ndarray): Gen image, BGR, [0, 255]
        img2 (np.ndarray): GT image, BGR, [0, 255]
        crop_border (int): pixels to crop.
    """
    # Convert to Y channel (float [0, 1])
    y1 = to_y_channel(img1)
    y2 = to_y_channel(img2)
    
    if crop_border > 0:
        y1 = y1[crop_border:-crop_border, crop_border:-crop_border]
        y2 = y2[crop_border:-crop_border, crop_border:-crop_border]
        
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse)) # using 1.0 as peak because y is [0,1]

def calculate_ssim(img1, img2, crop_border=0):
    """
    Calculate SSIM on Y channel.
    Args:
        img1 (np.ndarray): Gen image, BGR, [0, 255]
        img2 (np.ndarray): GT image, BGR, [0, 255]
        crop_border (int): pixels to crop.
    """
    y1 = to_y_channel(img1)
    y2 = to_y_channel(img2)
    
    if crop_border > 0:
        y1 = y1[crop_border:-crop_border, crop_border:-crop_border]
        y2 = y2[crop_border:-crop_border, crop_border:-crop_border]
        
    C1 = (0.01 * 1.0)**2
    C2 = (0.03 * 1.0)**2
    
    img1 = y1
    img2 = y2
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
