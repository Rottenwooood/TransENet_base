'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import argparse
import numpy as np
import cv2
import glob

PNG_DATASETS = {'AID', 'WHU-RS19', 'RSSCN7'}
DATASET_ROOTS = {
    'AID': '/root/autodl-tmp/TransENet_base/datasets/AID-dataset',
    'WHU-RS19': '/root/autodl-tmp/TransENet_base/datasets/WHU-RS19-dataset',
    'RSSCN7': '/root/autodl-tmp/TransENet_base/datasets/RSSCN7-dataset',
    'UCMerced': '/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_Gen', type=str, required=True, help='Generated images folder')
    parser.add_argument('--folder_GT', type=str, default=None, help='Ground-truth images folder')
    parser.add_argument('--dataset', type=str, default='UCMerced', choices=['UCMerced', 'AID', 'WHU-RS19', 'RSSCN7'],
                        help='Dataset name, used to infer default GT path and extension')
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale')
    parser.add_argument('--img_ext', type=str, default=None, help='Image extension, e.g. .tif or .png')
    parser.add_argument('--crop_border', type=int, default=None, help='Crop border for metric calculation')
    return parser.parse_args()


def main():
    args = parse_args()

    # Hardcoded configurations
    if args.folder_GT is not None:
        folder_GT = args.folder_GT
    else:
        dataset_root = DATASET_ROOTS[args.dataset]
        if args.dataset in PNG_DATASETS:
            folder_GT = os.path.join(dataset_root, 'test', 'HR')
        else:
            folder_GT = os.path.join(dataset_root, 'test', f'HR_x{args.scale}')
    folder_Gen = args.folder_Gen
    img_ext = args.img_ext or ('.png' if args.dataset in PNG_DATASETS else '.tif')
    crop_border = args.crop_border if args.crop_border is not None else args.scale
    suffix = ''
    test_Y = False

    print(f'GT folder: {folder_GT}')
    print(f'Gen folder: {folder_Gen}')
    print(f'Image extension: {img_ext}')

    PSNR_all = []
    SSIM_all = []

    # Support multiple extensions
    img_list = []
    search_exts = ['.png'] if args.dataset in PNG_DATASETS else ['.tif']
    for ext in search_exts:
        img_list.extend(glob.glob(os.path.join(folder_GT, f'*{ext}')))
    img_list = sorted(img_list)

    if not img_list:
        print(f'Error: No images found in {folder_GT}')
        return

    print(f'Found {len(img_list)} images')

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        im_GT = cv2.imread(img_path)
        im_Gen_path = os.path.join(folder_Gen, base_name + suffix + ext)

        if not os.path.exists(im_Gen_path):
            # Try with specified img_ext if not found
            im_Gen_path = os.path.join(folder_Gen, base_name + suffix + img_ext)

        if not os.path.exists(im_Gen_path):
            print(f'Warning: Generated image not found: {im_Gen_path}')
            continue

        im_GT = cv2.imread(img_path) / 255.
        im_Gen = cv2.imread(im_Gen_path) / 255.

        if im_GT is None or im_Gen is None:
            print(f'Warning: Failed to read images: {img_path} or {im_Gen_path}')
            continue

        if im_GT.shape != im_Gen.shape:
            print(f'Warning: Image size mismatch: {im_GT.shape} vs {im_Gen.shape}')
            continue

        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if crop_border == 0:
            cropped_GT = im_GT_in
            cropped_Gen = im_Gen_in
        else:
            if im_GT_in.ndim == 3:
                cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
            elif im_GT_in.ndim == 2:
                cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
                cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
            else:
                raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))


        # calculate PSNR and SSIM
        # PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
        PSNR = calculate_rgb_psnr(cropped_GT * 255, cropped_Gen * 255)

        SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
            i + 1, base_name, PSNR, SSIM))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)

    if not PSNR_all:
        print('Error: No valid images processed')
        return

    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all)))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_rgb_psnr(img1, img2):
    """calculate psnr among rgb channel, img1 and img2 have range [0, 255]
    """
    n_channels = np.ndim(img1)
    sum_psnr = 0
    for i in range(n_channels):
        this_psnr = calculate_psnr(img1[:,:,i], img2[:,:,i])
        sum_psnr += this_psnr
    return sum_psnr/n_channels

def ssim(img1, img2):
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


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(img1.shape[2]):
                ssims.append(ssim(img1[..., i], img2[..., i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
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


if __name__ == '__main__':
    main()
