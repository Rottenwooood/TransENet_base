import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from visualization_utils import (
    clamp_roi,
    compute_auto_roi,
    draw_box,
    ensure_dir,
    image_to_tensor,
    infer_hr_path,
    load_model_from_weights,
    load_rgb_image,
    normalize_map,
    save_rgb_image,
    tensor_to_image,
)


RED_RELEVANCE_CMAP = LinearSegmentedColormap.from_list(
    "red_relevance",
    [
        (0.0, "#ffffff"),
        (1.0, "#ff0000"),
    ],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Local Attribution Map (LAM) for SR models.")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument("--image", required=True, help="Path to LR image.")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--config", default=None, help="Optional config.txt path.")
    parser.add_argument("--model", default=None, help="Optional model name override.")
    parser.add_argument("--roi_size", type=int, default=48, help="ROI size on SR image.")
    parser.add_argument("--roi_x", type=int, default=None, help="ROI top-left x on SR image.")
    parser.add_argument("--roi_y", type=int, default=None, help="ROI top-left y on SR image.")
    parser.add_argument("--sigma", type=float, default=1.2, help="GaussianBlurPath initial sigma.")
    parser.add_argument("--fold", type=int, default=50, help="GaussianBlurPath interpolation steps.")
    parser.add_argument("--kernel_size", type=int, default=9, help="GaussianBlurPath kernel size.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend ratio for input image.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on.")
    return parser.parse_args()


def attr_grad(image, h, w, window=16):
    patch = image[:, :, h : h + window, w : w + window]
    return patch.mean()


def attribution_objective(attr_func, h, w, window=16):
    def calculate_objective(image):
        return attr_func(image, h, w, window=window)

    return calculate_objective


def isotropic_gaussian_kernel(size, sigma, epsilon=1e-5):
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * (sigma + epsilon) ** 2))
    return kernel / np.sum(kernel)


def GaussianBlurPath(sigma, fold, kernel_size=9):
    fold = max(1, int(fold))

    def path_interpolation_func(cv_numpy_image):
        h, w, c = cv_numpy_image.shape
        kernel_interpolation = np.zeros((fold + 1, kernel_size, kernel_size), dtype=np.float32)
        image_interpolation = np.zeros((fold, h, w, c), dtype=np.float32)
        lambda_derivative_interpolation = np.zeros((fold, h, w, c), dtype=np.float32)
        sigma_interpolation = np.linspace(sigma, 0, fold + 1)

        for i in range(fold + 1):
            kernel_interpolation[i] = isotropic_gaussian_kernel(kernel_size, sigma_interpolation[i])
        for i in range(fold):
            image_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, kernel_interpolation[i + 1])
            lambda_derivative_interpolation[i] = cv2.filter2D(
                cv_numpy_image,
                -1,
                (kernel_interpolation[i + 1] - kernel_interpolation[i]) * fold,
            )

        return (
            np.moveaxis(image_interpolation, 3, 1).astype(np.float32),
            np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32),
        )

    return path_interpolation_func


def Path_gradient(torch_image_numpy, model, attr_objective, path_interpolation_func, device):
    cv_numpy_image = torch_image_numpy[0].transpose(1, 2, 0)
    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(cv_numpy_image)
    grad_accumulate_list = np.zeros_like(lambda_derivative_interpolation, dtype=np.float32)
    result_list = []

    for i in range(image_interpolation.shape[0]):
        img_tensor = torch.from_numpy(image_interpolation[i : i + 1]).to(device)
        img_tensor.requires_grad_(True)
        result = model(img_tensor)
        target = attr_objective(result)
        model.zero_grad(set_to_none=True)
        target.backward()
        grad = img_tensor.grad.detach().cpu().numpy()[0]
        grad[np.isnan(grad)] = 0.0
        grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i]
        result_list.append(result.detach().cpu().numpy())

    result_numpy = np.asarray(result_list)
    return grad_accumulate_list, result_numpy, image_interpolation


def saliency_map(interpolated_grad_numpy, result_numpy):
    final_grad = interpolated_grad_numpy.mean(axis=0)
    return final_grad, result_numpy[-1]


def grad_abs_norm(grad):
    grad_2d = np.abs(grad.sum(axis=0))
    grad_max = grad_2d.max()
    if grad_max <= 0:
        return grad_2d
    return grad_2d / grad_max


def compute_lam(model, lr_tensor, roi_x, roi_y, roi_size, sigma, fold, kernel_size, device):
    attr_objective = attribution_objective(attr_grad, roi_y, roi_x, window=roi_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, kernel_size)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(
        lr_tensor.detach().cpu().numpy(),
        model,
        attr_objective,
        gaus_blur_path_func,
        device=device,
    )
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    return abs_normed_grad_numpy, result, interpolated_grad_numpy, result_numpy, interpolated_numpy


def save_figure(path, position, contribution, blend_abs, attribution, blend_kde, result_image):
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    panels = [
        (position, "Position"),
        (contribution, "Contribution"),
        (blend_abs, "Contribution + LR"),
        (attribution, "Attribution KDE"),
        (blend_kde, "Attribution + LR"),
        (result_image, "Result"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def relevance_to_rgb(relevance):
    relevance = normalize_map(relevance)
    rgba = RED_RELEVANCE_CMAP(relevance)
    return (rgba[:, :, :3] * 255.0).round().astype(np.uint8)


def vis_saliency_red(relevance, zoomin=4):
    image = relevance_to_rgb(relevance)
    h, w = image.shape[:2]
    return cv2.resize(image, (w * zoomin, h * zoomin), interpolation=cv2.INTER_NEAREST)


def vis_saliency_kde_red(relevance, zoomin=4):
    relevance = normalize_map(relevance)
    weights = relevance.reshape(-1).astype(np.float64)
    if weights.sum() <= 0:
        kde = relevance
    else:
        weights = weights + 1e-12
        y_grid, x_grid = np.mgrid[0 : relevance.shape[0] : 1, 0 : relevance.shape[1] : 1]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        pixels = np.vstack([x_grid.ravel(), y_grid.ravel()])
        try:
            kernel = stats.gaussian_kde(pixels, weights=weights)
            kde = np.reshape(kernel(positions).T, relevance.shape)
        except np.linalg.LinAlgError:
            kde = relevance
    image = relevance_to_rgb(kde)
    h, w = image.shape[:2]
    return cv2.resize(image, (w * zoomin, h * zoomin), interpolation=cv2.INTER_CUBIC)


def blend_with_lr(map_rgb, lr_image, alpha):
    lr_resized = cv2.resize(lr_image, (map_rgb.shape[1], map_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(map_rgb.astype(np.float32) * (1.0 - alpha) + lr_resized.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def save_relevance_map(path, relevance):
    image = relevance_to_rgb(relevance)
    save_rgb_image(path, image)


def result_to_image(result):
    result_tensor = torch.from_numpy(result[0])
    return tensor_to_image(result_tensor.unsqueeze(0))


def resize_to_match(image, reference):
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, model_args, _ = load_model_from_weights(
        weights_path=args.weights,
        model_name=args.model,
        config_path=args.config,
        device=device,
    )

    ensure_dir(args.output_dir)

    lr_image = load_rgb_image(args.image)
    lr_tensor = image_to_tensor(lr_image, device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_image = tensor_to_image(sr_tensor)

    hr_path = infer_hr_path(args.image, model_args.scale[0])
    hr_image = load_rgb_image(hr_path) if hr_path else None

    if args.roi_x is None or args.roi_y is None:
        roi_x, roi_y, roi_size = compute_auto_roi(sr_image, hr_image, args.roi_size)
    else:
        h, w = sr_image.shape[:2]
        roi_x, roi_y, roi_size = clamp_roi(args.roi_x, args.roi_y, args.roi_size, w, h)

    abs_normed_grad_numpy, result, interpolated_grad_numpy, result_numpy, interpolated_numpy = compute_lam(
        model=model,
        lr_tensor=lr_tensor,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_size=roi_size,
        sigma=args.sigma,
        fold=args.fold,
        kernel_size=args.kernel_size,
        device=device,
    )
    result_image = result_to_image(result)

    scale = model_args.scale[0]
    lr_roi_x = roi_x // scale
    lr_roi_y = roi_y // scale
    lr_roi_size = max(1, roi_size // scale)

    position_lr = draw_box(lr_image, lr_roi_x, lr_roi_y, lr_roi_size, color=(255, 0, 0))
    position = resize_to_match(position_lr, result_image)
    contribution = vis_saliency_red(abs_normed_grad_numpy, zoomin=scale)
    attribution = vis_saliency_kde_red(abs_normed_grad_numpy, zoomin=scale)
    blend_abs = blend_with_lr(contribution, lr_image, args.alpha)
    blend_kde = blend_with_lr(attribution, lr_image, args.alpha)

    basename = os.path.splitext(os.path.basename(args.image))[0]
    prefix = os.path.join(args.output_dir, basename)

    save_rgb_image(prefix + "_position.png", position)
    save_rgb_image(prefix + "_contribution.png", contribution)
    save_rgb_image(prefix + "_attribution.png", attribution)
    save_rgb_image(prefix + "_blend_abs.png", blend_abs)
    save_rgb_image(prefix + "_blend_kde.png", blend_kde)
    save_rgb_image(prefix + "_result.png", result_image)
    np.save(prefix + "_abs_normed_grad.npy", abs_normed_grad_numpy)
    np.save(prefix + "_interpolated_grad.npy", interpolated_grad_numpy)
    save_figure(prefix + "_lam_summary.png", position, contribution, blend_abs, attribution, blend_kde, result_image)

    print(f"Saved LAM outputs to: {args.output_dir}")
    print(f"ROI on SR image: x={roi_x}, y={roi_y}, size={roi_size}")
    print(f"GaussianBlurPath: sigma={args.sigma}, fold={args.fold}, kernel_size={args.kernel_size}")
    if hr_path:
        print(f"Reference HR image: {hr_path}")
    else:
        print("Reference HR image: not found, auto ROI used SR center or SR-vs-HR unavailable.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
