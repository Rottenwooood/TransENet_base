import sys
import pytest
import numpy as np
import cv2
import math
from pathlib import Path

# Add codes to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import calculate_psnr as new_psnr
from utils.metrics import calculate_ssim as new_ssim

# --- Legacy Logic (Copied from codes/legacy/metric_scripts/calculate_PSNR_SSIM.py) ---
def legacy_bgr2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def legacy_calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def legacy_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
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

# --- Tests ---

def test_psnr_consistency():
    """Verify PSNR matches legacy implementation on Y channel."""
    np.random.seed(42)
    
    # Generate random BGR images [0, 255] uint8
    img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Legacy Calculation
    # 1. Convert to float [0, 1]
    im1_f = img1.astype(np.float32) / 255.
    im2_f = img2.astype(np.float32) / 255.
    
    # 2. Convert to Y
    y1_legacy = legacy_bgr2ycbcr(im1_f)
    y2_legacy = legacy_bgr2ycbcr(im2_f)
    
    # 3. Calculate PSNR (input scaled to 0-255)
    psnr_legacy = legacy_calculate_psnr(y1_legacy * 255, y2_legacy * 255)
    
    # New Calculation
    psnr_new = new_psnr(img1, img2, crop_border=0)
    
    print(f"Legacy PSNR: {psnr_legacy}")
    print(f"New PSNR: {psnr_new}")
    
    assert np.isclose(psnr_legacy, psnr_new, atol=1e-4)

def test_ssim_consistency():
    """Verify SSIM matches legacy implementation on Y channel."""
    np.random.seed(42)
    
    img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Legacy
    im1_f = img1.astype(np.float32) / 255.
    im2_f = img2.astype(np.float32) / 255.
    y1_legacy = legacy_bgr2ycbcr(im1_f)
    y2_legacy = legacy_bgr2ycbcr(im2_f)
    
    # Legacy SSIM with 0 border crop (slicing handled inside legacy_ssim is [5:-5] valid padding, 
    # but the script did explicit cropping before passing to ssim)
    
    # The script logic:
    # cropped_GT = im_GT_in[crop_border:-crop_border, ...]
    # SSIM = calculate_ssim(cropped_GT * 255, ...)
    # calculate_ssim calls ssim
    # ssim does generic valid padding [5:-5]
    
    # Our new_ssim does:
    # y1 = to_y_channel(img1)
    # y1 = y1[crop_border:-crop_border] (if > 0)
    # ...
    # mu1 = filter2D(...)[5:-5, 5:-5]
    
    # So if we pass crop_border=0 to both, they should match.
    
    ssim_legacy = legacy_ssim(y1_legacy * 255, y2_legacy * 255)
    ssim_new = new_ssim(img1, img2, crop_border=0)
    
    print(f"Legacy SSIM: {ssim_legacy}")
    print(f"New SSIM: {ssim_new}")
    
    assert np.isclose(ssim_legacy, ssim_new, atol=1e-4)

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    sys.exit(pytest.main(["-v", __file__]))
