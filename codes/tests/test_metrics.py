import sys
import os
import numpy as np
import pytest
import cv2
from pathlib import Path

# Add codes to path
sys.path.append(str(Path(__file__).parent.parent))

import utils
# We need to import calculate_psnr from batch_deploy_psnr
# But batch_deploy_psnr is a script, might execute main if not careful.
# It has if __name__ == '__main__': main(), so it's safe to import.
from batch_deploy_psnr import calculate_psnr, calculate_ssim

def test_bgr2ycbcr_values():
    """Test BGR to YCbCr conversion values against known standards"""
    # Pure Blue (255, 0, 0) in BGR -> (0, 0, 255) in RGB
    # MATLAB: Y = 16 + 65.481*R/255 + 128.553*G/255 + 24.966*B/255
    # For Blue (R=0, G=0, B=1): Y = 16 + 24.966 = 40.966 -> 41
    
    bgr_blue = np.array([[[255, 0, 0]]], dtype=np.uint8)
    y_blue = utils.bgr2ycbcr(bgr_blue, only_y=True)
    
    # Cast to float to avoid uint8 overflow/underflow in assertion
    y_val = float(y_blue[0,0])
    assert abs(y_val - 41) <= 1.0

    # Pure White (255, 255, 255)
    # Y = 16 + (24.966 + 128.553 + 65.481) = 16 + 219 = 235
    bgr_white = np.array([[[255, 255, 255]]], dtype=np.uint8)
    y_white = utils.bgr2ycbcr(bgr_white, only_y=True)
    assert abs(y_white[0,0] - 235) <= 1.0
    
    # Pure Black (0, 0, 0)
    # Y = 16
    bgr_black = np.array([[[0, 0, 0]]], dtype=np.uint8)
    y_black = utils.bgr2ycbcr(bgr_black, only_y=True)
    assert abs(y_black[0,0] - 16) <= 1.0

def test_psnr_calculation():
    """Test PSNR calculation logic"""
    # Identical images should have Infinite PSNR
    img1 = np.ones((10, 10, 3), dtype=np.float64) * 255.0
    img2 = np.ones((10, 10, 3), dtype=np.float64) * 255.0
    
    psnr = calculate_psnr(img1, img2)
    assert psnr == float('inf')
    
    # Known difference
    # MSE = 10^2 = 100
    # PSNR = 20 * log10(255 / sqrt(100)) = 20 * log10(25.5) ~= 28.13 dB
    img3 = img1 + 10.0
    psnr_diff = calculate_psnr(img1, img3)
    assert abs(psnr_diff - 28.13) < 0.1

def test_shave_border_logic_simulation():
    """Simulate the border shaving logic used in deploy script"""
    scale = 4
    h, w = 20, 20
    img = np.zeros((h, w), dtype=np.float32)
    
    # Fill center with 1, border with 0
    img[scale:-scale, scale:-scale] = 1.0
    
    cropped = img[scale:-scale, scale:-scale]
    
    assert cropped.shape == (h - 2*scale, w - 2*scale)
    assert np.all(cropped == 1.0)
    
if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
