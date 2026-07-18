"""
Calculate SCC and SAM for remote-sensing SR results.

The implementation follows the sewar.full_ref scc() and sam() metrics used by
HAUNet_RSISR.
"""
import argparse
import glob
import multiprocessing as mp
import os

import cv2
import numpy as np

PNG_DATASETS = {'AID', 'WHU-RS19', 'RSSCN7'}
DATASET_ROOTS = {
    'AID': '/root/autodl-tmp/TransENet_base/datasets/AID-dataset',
    'WHU-RS19': '/root/autodl-tmp/TransENet_base/datasets/WHU-RS19-dataset',
    'RSSCN7': '/root/autodl-tmp/TransENet_base/datasets/RSSCN7-dataset',
    'UCMerced': '/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_Gen', type=str, required=True, help='Generated images folder or experiment result root')
    parser.add_argument('--folder_GT', type=str, default=None, help='Ground-truth images folder')
    parser.add_argument('--dataset', type=str, default='UCMerced', choices=['UCMerced', 'AID', 'WHU-RS19', 'RSSCN7'],
                        help='Dataset name, used to infer default GT path and extension')
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale')
    parser.add_argument('--img_ext', type=str, default=None, help='Image extension, e.g. .tif or .png')
    parser.add_argument('--crop_border', type=int, default=None, help='Crop border for metric calculation')
    parser.add_argument('--eps', type=float, default=1e-12, help='Kept for backward-compatible CLI parsing')
    parser.add_argument('--n_threads', type=int, default=1, help='Number of worker processes for image-level parallelism')
    return parser.parse_args()


def resolve_gt_folder(args):
    if args.folder_GT is not None:
        return args.folder_GT

    dataset_root = DATASET_ROOTS[args.dataset]
    if args.dataset in PNG_DATASETS:
        return os.path.join(dataset_root, 'test', 'HR')
    return os.path.join(dataset_root, 'test', f'HR_x{args.scale}')


def resolve_gen_folder(folder_gen, scale, img_ext):
    folder_gen = os.path.abspath(folder_gen)

    if not os.path.isdir(folder_gen):
        raise FileNotFoundError(f'Generated folder does not exist: {folder_gen}')

    direct_matches = glob.glob(os.path.join(folder_gen, f'*{img_ext}'))
    if direct_matches:
        return folder_gen

    nested_scale_dir = os.path.join(folder_gen, f'x{scale}')
    if os.path.isdir(nested_scale_dir):
        nested_matches = glob.glob(os.path.join(nested_scale_dir, f'*{img_ext}'))
        if nested_matches:
            return nested_scale_dir

    raise FileNotFoundError(
        f'No generated images found under {folder_gen} or {nested_scale_dir} with extension {img_ext}'
    )


def _filter2(img, fltr, mode='same'):
    if mode != 'same':
        raise ValueError(f'Unsupported filter mode: {mode}')
    return cv2.filter2D(img, -1, np.asarray(fltr, dtype=np.float32), borderType=cv2.BORDER_CONSTANT)


def _uniform_window(ws):
    return np.ones((ws, ws), dtype=np.float32) / (ws ** 2)


def _get_sums(gt, pred, win, mode='same'):
    mu_gt = _filter2(gt, win, mode)
    mu_pred = _filter2(pred, win, mode)
    return mu_gt * mu_gt, mu_pred * mu_pred, mu_gt * mu_pred


def _get_sigmas(gt, pred, win, mode='same'):
    gt_sum_sq, pred_sum_sq, gt_pred_sum_mul = _get_sums(gt, pred, win, mode)
    return (
        _filter2(gt * gt, win, mode) - gt_sum_sq,
        _filter2(pred * pred, win, mode) - pred_sum_sq,
        _filter2(gt * pred, win, mode) - gt_pred_sum_mul,
    )


def calculate_scc(img1, img2, win=None, ws=8):
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same shape for SCC.')

    if win is None:
        win = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    win = np.asarray(win, dtype=np.float32)

    gt_hp = cv2.filter2D(img1.astype(np.float32), -1, win, borderType=cv2.BORDER_REFLECT)
    pred_hp = cv2.filter2D(img2.astype(np.float32), -1, win, borderType=cv2.BORDER_REFLECT)
    sigma_gt_sq, sigma_pred_sq, sigma_gt_pred = _get_sigmas(gt_hp, pred_hp, _uniform_window(ws))

    sigma_gt_sq[sigma_gt_sq < 0] = 0
    sigma_pred_sq[sigma_pred_sq < 0] = 0

    den = np.sqrt(sigma_gt_sq) * np.sqrt(sigma_pred_sq)
    zero_den = den == 0
    den[zero_den] = 1
    scc_map = sigma_gt_pred / den
    scc_map[zero_den] = 0
    return float(np.mean(scc_map))


def calculate_sam(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same shape for SAM.')

    if img1.ndim != 3 or img1.shape[2] < 2:
        raise ValueError('SAM expects multi-channel images with shape HxWxC.')

    x = img1.reshape(-1, img1.shape[2]).astype(np.float64)
    y = img2.reshape(-1, img2.shape[2]).astype(np.float64)

    sam_angles = np.zeros(x.shape[1], dtype=np.float64)
    for channel_idx in range(x.shape[1]):
        denom = np.linalg.norm(x[:, channel_idx]) * np.linalg.norm(y[:, channel_idx])
        val = np.clip(np.dot(x[:, channel_idx], y[:, channel_idx]) / denom, -1, 1)
        sam_angles[channel_idx] = np.arccos(val)
    return float(np.mean(sam_angles))


def crop_image(img, crop_border):
    if crop_border == 0:
        return img

    if img.ndim == 3:
        return img[crop_border:-crop_border, crop_border:-crop_border, :]
    if img.ndim == 2:
        return img[crop_border:-crop_border, crop_border:-crop_border]
    raise ValueError(f'Wrong image dimension: {img.ndim}. Should be 2 or 3.')


def process_image(task):
    idx, gt_path, folder_gen, img_ext, crop_border = task
    base_name = os.path.splitext(os.path.basename(gt_path))[0]
    gen_path = os.path.join(folder_gen, base_name + img_ext)

    if not os.path.exists(gen_path):
        return idx, base_name, None, None, f'Warning: Generated image not found: {gen_path}'

    gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    gen = cv2.imread(gen_path, cv2.IMREAD_COLOR)
    if gt is None or gen is None:
        return idx, base_name, None, None, f'Warning: Failed to read images: {gt_path} or {gen_path}'

    if gt.shape != gen.shape:
        return idx, base_name, None, None, f'Warning: Image size mismatch: {gt.shape} vs {gen.shape}'

    gt = crop_image(gt.astype(np.float64) / 255.0, crop_border)
    gen = crop_image(gen.astype(np.float64) / 255.0, crop_border)

    scc = calculate_scc(gt * 255.0, gen * 255.0)
    sam = calculate_sam(gt * 255.0, gen * 255.0)
    return idx, base_name, scc, sam, None


def main():
    args = parse_args()
    cv2.setNumThreads(0)
    img_ext = args.img_ext or ('.png' if args.dataset in PNG_DATASETS else '.tif')
    crop_border = args.crop_border if args.crop_border is not None else args.scale
    folder_gt = resolve_gt_folder(args)
    folder_gen = resolve_gen_folder(args.folder_Gen, args.scale, img_ext)
    n_threads = max(1, args.n_threads)

    print(f'GT folder: {folder_gt}')
    print(f'Gen folder: {folder_gen}')
    print(f'Image extension: {img_ext}')
    print(f'Crop border: {crop_border}')
    print(f'Workers: {n_threads}')

    gt_img_list = sorted(glob.glob(os.path.join(folder_gt, f'*{img_ext}')))
    if not gt_img_list:
        print(f'Error: No images found in {folder_gt}')
        return

    print(f'Found {len(gt_img_list)} images')

    scc_all = []
    sam_all = []
    tasks = [(idx, gt_path, folder_gen, img_ext, crop_border) for idx, gt_path in enumerate(gt_img_list)]
    if n_threads == 1:
        results = map(process_image, tasks)
    else:
        with mp.Pool(processes=n_threads) as pool:
            results = pool.imap_unordered(process_image, tasks)

    for idx, base_name, scc, sam, warning in results:
        if warning is not None:
            print(warning)
            continue
        print('{:3d} - {:25}. \tSCC: {:.6f}, \tSAM: {:.6f}'.format(
            idx + 1, base_name, scc, sam))
        scc_all.append(scc)
        sam_all.append(sam)

    if not scc_all:
        print('Error: No valid images processed')
        return

    print('Average: SCC: {:.6f}, SAM: {:.6f}'.format(
        sum(scc_all) / len(scc_all),
        sum(sam_all) / len(sam_all)))


if __name__ == '__main__':
    main()
