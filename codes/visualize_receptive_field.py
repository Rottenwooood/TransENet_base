import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from visualization_utils import (
    ensure_dir,
    image_to_tensor,
    load_model_from_weights,
    load_rgb_image,
    normalize_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize effective receptive field (ERF) for SR models.")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--image", default=None, help="Single LR image path.")
    parser.add_argument("--image_dir", default=None, help="LR image directory for ERF accumulation.")
    parser.add_argument("--config", default=None, help="Optional config.txt path.")
    parser.add_argument("--model", default=None, help="Optional model name override.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on.")
    parser.add_argument("--num_images", type=int, default=32, help="How many images to accumulate.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index after sorting image list.")
    parser.add_argument("--reduction", choices=["sum", "mean"], default="sum", help="How to aggregate gradients over images.")
    parser.add_argument("--center_x", type=int, default=None, help="Target x on SR image.")
    parser.add_argument("--center_y", type=int, default=None, help="Target y on SR image.")
    parser.add_argument("--power", type=float, default=0.25, help="Power normalization exponent for display.")
    parser.add_argument("--save_each", action="store_true", help="Save per-image gradient maps too.")
    return parser.parse_args()


def resolve_image_list(args):
    if args.image:
        return [args.image]

    if args.image_dir:
        patterns = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp"]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(args.image_dir, pattern)))
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"No images found in: {args.image_dir}")
        return paths[args.start_index : args.start_index + args.num_images]

    raise ValueError("Pass either --image or --image_dir.")


def mark_center(image_rgb, center_x, center_y, radius=4):
    canvas = image_rgb.copy()
    cv2.circle(canvas, (center_x, center_y), radius, (255, 0, 0), -1)
    return canvas


def accumulate_erf(model, image_paths, device, center_xy=None, save_each_dir=None):
    grad_sum = None
    sr_reference = None
    center_x = None
    center_y = None

    for idx, image_path in enumerate(image_paths):
        image = load_rgb_image(image_path)
        tensor = image_to_tensor(image, device)
        tensor = tensor.requires_grad_(True)

        sr = model(tensor)
        if sr_reference is None:
            sr_reference = sr.detach().cpu()
            _, _, h_sr, w_sr = sr.shape
            center_x = center_xy[0] if center_xy and center_xy[0] is not None else w_sr // 2
            center_y = center_xy[1] if center_xy and center_xy[1] is not None else h_sr // 2
            center_x = int(np.clip(center_x, 0, w_sr - 1))
            center_y = int(np.clip(center_y, 0, h_sr - 1))

        score = sr[:, :, center_y, center_x].sum()
        model.zero_grad(set_to_none=True)
        if tensor.grad is not None:
            tensor.grad.zero_()
        score.backward()

        grad = tensor.grad.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()
        if grad_sum is None:
            grad_sum = grad
        else:
            grad_sum += grad

        if save_each_dir:
            heat = normalize_map(grad)
            heat = np.power(heat, 0.25)
            plt.imsave(os.path.join(save_each_dir, f"{idx:03d}_{os.path.splitext(os.path.basename(image_path))[0]}_erf.png"), heat, cmap="YlGn")

    return grad_sum, sr_reference, center_x, center_y


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, _, _ = load_model_from_weights(
        weights_path=args.weights,
        model_name=args.model,
        config_path=args.config,
        device=device,
    )

    ensure_dir(args.output_dir)
    image_paths = resolve_image_list(args)
    each_dir = os.path.join(args.output_dir, "each") if args.save_each else None
    if each_dir:
        ensure_dir(each_dir)

    grad_map, sr_reference, center_x, center_y = accumulate_erf(
        model=model,
        image_paths=image_paths,
        device=device,
        center_xy=(args.center_x, args.center_y),
        save_each_dir=each_dir,
    )

    if args.reduction == "mean":
        grad_map = grad_map / max(1, len(image_paths))

    grad_map = normalize_map(grad_map)
    grad_map = np.power(grad_map, args.power)
    grad_map = normalize_map(grad_map)

    sr_reference = sr_reference.squeeze(0).permute(1, 2, 0).numpy()
    sr_reference = np.clip(sr_reference, 0, 1)
    sr_reference = (sr_reference * 255.0).round().astype(np.uint8)
    sr_marked = mark_center(sr_reference, center_x, center_y)

    plt.imsave(os.path.join(args.output_dir, "erf_heatmap.png"), grad_map, cmap="YlGn")
    plt.imsave(os.path.join(args.output_dir, "erf_heatmap_jet.png"), grad_map, cmap="jet")
    cv2.imwrite(
        os.path.join(args.output_dir, "erf_sr_reference.png"),
        cv2.cvtColor(sr_marked, cv2.COLOR_RGB2BGR),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(sr_marked)
    axes[0].set_title("SR Reference")
    axes[1].imshow(grad_map, cmap="YlGn")
    axes[1].set_title("Effective Receptive Field")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "erf_summary.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ERF outputs to: {args.output_dir}")
    print(f"Used {len(image_paths)} images")
    print(f"Target SR coordinate: x={center_x}, y={center_y}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
