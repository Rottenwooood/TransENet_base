import ast
import os
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")


def parse_experiment_config(config_path):
    config = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.count(":") == 0:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_value(value.strip())
    return config


def _parse_value(value):
    if value in {"True", "False"}:
        return value == "True"
    if value == "None":
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def infer_config_path(weights_path, config_path=None):
    if config_path:
        return config_path
    weights_path = Path(weights_path).resolve()
    experiment_dir = weights_path.parent.parent
    inferred = experiment_dir / "config.txt"
    if not inferred.exists():
        raise FileNotFoundError(f"Could not infer config.txt from weights: {weights_path}")
    return str(inferred)


def build_model_args(model_name, config):
    defaults = {
        "cpu": False,
        "n_GPUs": 1,
        "precision": "single",
        "self_ensemble": False,
        "chop": False,
        "save_models": False,
        "print_model": False,
        "resume": 0,
        "pre_train": ".",
        "n_colors": 3,
        "scale": [4],
        "symunet_pretrain_width": 32,
        "symunet_pretrain_middle_blk_num": 1,
        "symunet_pretrain_enc_blk_nums": [4, 6],
        "symunet_pretrain_dec_blk_nums": [6, 4],
        "symunet_pretrain_ffn_expansion_factor": 2.66,
        "symunet_pretrain_bias": False,
        "symunet_pretrain_layer_norm_type": "WithBias",
        "symunet_pretrain_restormer_heads": [1, 2, 4],
        "symunet_pretrain_restormer_middle_heads": 8,
        "symunet_pretrain_strip_k1": 1,
        "symunet_pretrain_strip_k2": 47,
        "symunet_pretrain_mab2_kernel_sizes": [7, 11],
        "symunet_pretrain_mab2_dilations": [5, 3],
        "cga_paper_split_size": [8, 32],
    }
    defaults.update(config)
    defaults["model"] = model_name
    return SimpleNamespace(**defaults)


def extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "params", "params_ema", "net", "network"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break

    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload is not a state dict.")

    cleaned = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_model_from_weights(weights_path, model_name=None, config_path=None, device=None):
    config_path = infer_config_path(weights_path, config_path)
    config = parse_experiment_config(config_path)
    model_name = model_name or config.get("model")
    if not model_name:
        raise ValueError("Model name is missing. Pass --model or provide a config.txt with model entry.")

    args = build_model_args(model_name, config)
    module = import_module("model." + model_name.lower())
    model = module.make_model(args)

    state = torch.load(weights_path, map_location="cpu")
    state_dict = extract_state_dict(state)
    model.load_state_dict(state_dict, strict=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, args, config


def load_rgb_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_tensor(image_rgb, device):
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float()
    tensor = tensor.unsqueeze(0) / 255.0
    return tensor.to(device)


def tensor_to_image(tensor):
    array = tensor.detach().cpu().squeeze(0).clamp(0, 1).numpy()
    array = np.transpose(array, (1, 2, 0))
    return (array * 255.0).round().astype(np.uint8)


def infer_hr_path(lr_path, scale):
    lr_path = Path(lr_path).resolve()
    token = f"LR_x{scale}"
    replacement = f"HR_x{scale}"
    if token not in str(lr_path):
        return None
    hr_path = Path(str(lr_path).replace(token, replacement))
    return str(hr_path) if hr_path.exists() else None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_map(array):
    array = np.asarray(array, dtype=np.float32)
    array = array - array.min()
    max_value = array.max()
    if max_value > 0:
        array = array / max_value
    return array


def overlay_heatmap(base_rgb, heatmap, alpha=0.55):
    heatmap = normalize_map(heatmap)
    heatmap_u8 = np.uint8(heatmap * 255.0)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return np.clip(base_rgb.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def upscale_heatmap(heatmap, size_hw):
    target_h, target_w = size_hw
    return cv2.resize(heatmap.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def clamp_roi(x, y, size, width, height):
    size = max(1, min(size, width, height))
    x = max(0, min(x, width - size))
    y = max(0, min(y, height - size))
    return int(x), int(y), int(size)


def compute_auto_roi(sr_image, hr_image, roi_size):
    if hr_image is None:
        h, w = sr_image.shape[:2]
        return clamp_roi((w - roi_size) // 2, (h - roi_size) // 2, roi_size, w, h)

    diff = np.mean(np.abs(sr_image.astype(np.float32) - hr_image.astype(np.float32)), axis=2)
    h, w = diff.shape
    roi_size = max(1, min(roi_size, h, w))
    best_score = -1.0
    best_xy = (0, 0)
    stride = max(1, roi_size // 4)
    for y in range(0, h - roi_size + 1, stride):
        for x in range(0, w - roi_size + 1, stride):
            score = float(diff[y : y + roi_size, x : x + roi_size].mean())
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return clamp_roi(best_xy[0], best_xy[1], roi_size, w, h)


def draw_box(image_rgb, x, y, size, color=(255, 0, 0), thickness=1):
    canvas = image_rgb.copy()
    cv2.rectangle(canvas, (x, y), (x + size, y + size), color, thickness)
    return canvas


def save_rgb_image(path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)
