# LAM 与感受野图使用说明

以下命令默认使用 `conda` 环境 `transenet-pren`。

## 1. LAM 图

```bash
conda run -n transenet-pren python codes/visualize_lam.py \
  --weights /root/autodl-tmp/TransENet_base/experiment/s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x3_w32_20260429_071942/model/model_best.pt \
  --image /root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset/test/LR_x3/agricultural50.tif \
  --output_dir /root/autodl-tmp/TransENet_base/experiment/vis_lam \
  --roi_x 120 --roi_y 90 --roi_size 48
```

可选参数：

- `--roi_size 48`：SR 图上 ROI 的边长，这个区域就是反传归因目标。
- `--roi_x 120 --roi_y 90`：手动指定 ROI 左上角。
- `--sigma 1.2`：官方 GaussianBlurPath 初始 sigma。
- `--fold 50`：官方 GaussianBlurPath 路径采样步数。
- `--kernel_size 9`：官方 GaussianBlurPath 高斯核尺寸。
- `--alpha 0.5`：热图和 LR 图混合显示的 LR 权重。
- `--explain residual`：解释 `SR - bicubic(LR)` 残差响应；默认 `--explain sr` 解释最终 SR 输出。

## 2. 感受野图（ERF）

```bash
conda run -n transenet-pren python codes/visualize_receptive_field.py \
  --weights /root/autodl-tmp/TransENet_base/experiment/s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x3_w32_20260429_071942/model/model_best.pt \
  --image_dir /root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset/test/LR_x3 \
  --num_images 32 \
  --output_dir /root/autodl-tmp/TransENet_base/experiment/vis_erf
```

可选参数：

- `--center_x 180 --center_y 180`：指定 SR 输出图上的目标像素坐标。
- `--power 0.25`：控制显示时的幂次归一化，常用来增强弱响应区域可见性。
- `--reduction mean`：对多图梯度取平均。

## 3. 输出内容

LAM 会输出：

- `*_position.png`
- `*_contribution.png`
- `*_attribution.png`
- `*_blend_abs.png`
- `*_blend_kde.png`
- `*_result.png`
- `*_abs_normed_grad.npy`
- `*_interpolated_grad.npy`
- `*_lam_summary.png`

感受野图会输出：

- `erf_sr_reference.png`
- `erf_heatmap.png`
- `erf_heatmap_jet.png`
- `erf_summary.png`

## 4. 说明

- 脚本会优先从权重所在实验目录自动读取 `config.txt`，自动恢复模型结构参数。
- 如果 LR 图路径里包含 `LR_x3` 这类目录名，脚本会自动尝试匹配对应的 `HR_x3` 参考图。
- 当前 LAM 实现对齐官方流程：`attribution_objective(attr_grad, h, w, window)`、`GaussianBlurPath(sigma, fold, kernel_size)`、`Path_gradient`、`saliency_map`、`grad_abs_norm`、绝对 saliency 与 KDE saliency 可视化。
- 如果最终 SR 的 LAM 过度集中在 ROI 对应局部，可使用 `--explain residual` 减弱 bicubic/残差捷径的局部主导效应。
- Contribution 和 Attribution 图上不额外画 OpenCV 标记，背景为纯白，红色深浅按归一化相关性线性变化。
- 由于该模型输入是 LR，原始归因值定义在 LR 网格上；`*_contribution.png` 和 `*_attribution.png` 是按照超分倍率放大的显示图，原始数值请以 `*_abs_normed_grad.npy` 为准。
- 当前 ERF 实现采用“固定 SR 中心像素，对多张 LR 图累计输入梯度”的方式，更接近论文里常见的 ERF 统计图。
