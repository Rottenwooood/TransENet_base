# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch-based deep learning repository for remote sensing image super-resolution. The main model is **TransENet** (Transformer-based Multi-Stage Enhancement Network), which has been enhanced with **SymUNet** and its variants. The project supports multiple datasets (UCMerced, AID, DIV2K) and various super-resolution scales (2x, 3x, 4x, 8x+).

## Environment Setup

**Conda Environment:**
```bash
conda activate symunet
```

**Python Version:** 3.10+

**Install Dependencies:**
```bash
cd codes
pip install -r ../requirements.txt
```

## Common Commands

### Training

**Basic Training:**
```bash
cd codes

# Train TransENet on UCMerced (x4 scale)
python demo_train.py --model=TRANSENET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=TRANSENETx4_UCMerced

# Train TransENet on UCMerced (x3 scale)
python demo_train.py --model=TRANSENET --dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=TRANSENETx3_UCMerced

# Train TransENet on UCMerced (x2 scale)
python demo_train.py --model=TRANSENET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=TRANSENETx2_UCMerced
```

**SymUNet Variants Training:**

*SymUNet (Standard):*
```bash
python demo_train.py --model=SYMUNET --dataset=UCMerced --scale=4 --save=SYMUNETx4
```

*SymUNet-Pretrain (Pre-upsampling):*
```bash
python demo_train.py \
    --model=SYMUNET_PRETRAIN \
    --dataset=UCMerced \
    --scale=4 \
    --symunet_pretrain_width=48 \
    --symunet_pretrain_enc_blk_nums=4,6,6 \
    --symunet_pretrain_dec_blk_nums=6,6,4 \
    --symunet_pretrain_restormer_heads=1,2,4 \
    --epochs=500 \
    --batch_size=4 \
    --save=SYMUNET_PRETRAIN_x4
```

*SymUNet-Posttrain (Post-upsampling):*
```bash
python demo_train.py \
    --model=SYMUNET_POSTTRAIN \
    --dataset=UCMerced \
    --scale=4 \
    --symunet_posttrain_width=64 \
    --symunet_posttrain_enc_blk_nums=2,2,2 \
    --symunet_posttrain_dec_blk_nums=2,2,2 \
    --symunet_posttrain_restormer_heads=1,2,4 \
    --epochs=500 \
    --batch_size=16 \
    --save=SYMUNET_POSTTRAIN_x4
```

**Resume Training:**
```bash
python demo_train.py --resume=1 --save=EXPERIMENT_NAME
```

**Test During Training:**
```bash
python demo_train.py --test_only --save=EXPERIMENT_NAME
```

### Testing/Inference

**Deploy Model:**
```bash
cd codes

# Test TransENet
python demo_deploy.py --model=TRANSENET --scale=4

# Test SymUNet variants
python demo_deploy.py --model=SYMUNET --scale=4
python demo_deploy.py --model=SYMUNET_PRETRAIN --scale=4
python demo_deploy.py --model=SYMUNET_POSTTRAIN --scale=4
```

**Model Performance Analysis:**
```bash
python demo_deploy.py --model=TRANSENET --scale=4 --test_performance
```
This calculates params, FLOPs, memory usage, and inference time.

**Calculate PSNR/SSIM:**
```bash
cd codes/metric_scripts
python calculate_PSNR_SSIM.py
```

### Quick Test Scripts

**Test SymUNet Model:**
```bash
cd codes
python test_symunet.py
```

**Test SymUNet SR Versions:**
```bash
cd codes
python test_symunet_sr_versions.py
```

**Training Examples:**
```bash
cd codes
python train_symunet_example.py
python train_symunet_sr_examples.py
```

## Project Architecture

```
TransENet_base/
├── codes/                          # Main source code directory
│   ├── model/                      # Model definitions
│   │   ├── transenet.py           # TransENet model (main contribution)
│   │   ├── symunet.py            # Standard SymUNet model
│   │   ├── symunet_pretrain.py   # Pre-upsampling variant
│   │   ├── symunet_posttrain.py  # Post-upsampling variant
│   │   ├── transformer.py        # Transformer components
│   │   ├── basic.py              # Basic CNN blocks
│   │   └── ...                   # Other SR models (SRCNN, VDSR, etc.)
│   ├── data/                      # Data loading and preprocessing
│   │   ├── aid.py                # AID dataset
│   │   ├── ucmerced.py           # UCMerced dataset
│   │   ├── div2k.py              # DIV2K dataset
│   │   └── common.py             # Common data utilities
│   ├── loss/                      # Loss functions
│   │   ├── adversarial.py        # GAN loss
│   │   ├── vgg.py                # Perceptual loss
│   │   ├── fft.py                # FFT loss
│   │   └── ...
│   ├── metric_scripts/            # Evaluation metrics
│   ├── data_scripts/              # Data generation scripts
│   ├── demo_train.py             # Training entry point
│   ├── demo_deploy.py            # Testing/deployment entry point
│   ├── trainer.py                # Training loop
│   ├── option.py                 # Argument parsing
│   ├── utils.py                  # Utilities
│   └── imresize.py               # Image resizing
├── SymUNet/                       # SymUNet-specific components
│   └── basicsr/                   # BasicSR framework integration
├── SYMUNET_USAGE.md               # Detailed SymUNet usage guide
├── SYMUNET_SR_VERSIONS.md        # SymUNet SR variant documentation
└── requirements.txt              # Dependencies
```

### Key Components

**Training Pipeline (`demo_train.py`):**
1. Loads arguments from `option.py`
2. Creates dataloaders from `data/` module
3. Initializes model from `model/` module via dynamic import
4. Creates loss function from `loss/` module
5. Runs training loop in `trainer.py`

**Model Loading (`model/__init__.py`):**
- Uses Python's `importlib` to dynamically load model based on `--model` argument
- Supports: TRANSENET, SYMUNET, SYMUNET_PRETRAIN, SYMUNET_POSTTRAIN, SRCNN, VDSR, FSRCNN, etc.

**Checkpoint System (`utils.py`):**
- Automatic experiment logging to `../experiment/{save_name}/`
- Saves: model weights, optimizer state, training logs, loss curves
- Config saved to `config.txt`

## Model Variants

### TransENet
- Original transformer-based multi-stage enhancement network
- Uses transformer blocks for feature enhancement
- Standard upsampling at end

### SymUNet Variants

**Standard SymUNet (`symunet.py`):**
- Symmetric encoder-decoder architecture
- Reconstructor block for upsampling
- Balanced quality and efficiency

**SymUNet-Pretrain (`symunet_pretrain.py`):**
- Bicubic upsampling at input
- Processes in HR space throughout
- **Pros:** Better detail recovery, higher quality
- **Cons:** High memory usage, slower
- **Best for:** 2x, 4x SR with充足 resources

**SymUNet-Posttrain (`symunet_posttrain.py`):**
- Feature extraction in LR space
- PixelShuffle upsampling at end
- **Pros:** Memory efficient, faster, scalable
- **Cons:** Slightly lower quality
- **Best for:** 8x+, real-time, resource-constrained

## Datasets

**Supported Datasets:**
- **UCMerced:** Remote sensing land use dataset (default)
- **AID:** Another remote sensing dataset
- **DIV2K:** Standard super-resolution dataset

**Dataset Paths:** Configured in `codes/data/__init__.py` (lines 8, 15, 22)
- Can override via `--data_train` and `--data_val` arguments

**Data Format:**
- LR-HR image pairs
- HR generated via bicubic downsampling
- Extensions: `.img` (raw images), `.npy` (numpy arrays), `.pt` (PyTorch tensors)

## Important Configuration Options

**Core Training Args:**
- `--model`: Model name (TRANSENET, SYMUNET, SYMUNET_PRETRAIN, etc.)
- `--dataset`: Dataset name (UCMerced, AID, DIV2K)
- `--scale`: Super-resolution scale (2, 3, 4, 8)
- `--epochs`: Training epochs (default: 500)
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--patch_size`: Training patch size

**SymUNet-Specific Args:**
- `--symunet_width`: Base channel width (32, 64, 128)
- `--symunet_enc_blk_nums`: Encoder depths per stage (e.g., 2,2,2)
- `--symunet_dec_blk_nums`: Decoder depths per stage (e.g., 2,2,2)
- `--symunet_restormer_heads`: Attention heads per stage (e.g., 1,2,4)
- `--symunet_ffn_expansion_factor`: FFN expansion (default: 2.66)

**Output Control:**
- `--save`: Experiment name (creates `../experiment/{save_name}/`)
- `--save_models`: Save model checkpoints for all epochs
- `--test_metric`: Metric for best model selection (psnr, ssim)

## Performance Optimization

**Memory-Efficient Training:**
- Use `--chop` for memory-efficient forward pass
- Reduce `--batch_size` if OOM errors occur
- Use SymUNet-Posttrain for large scales

**Multi-GPU Training:**
- Set `--n_GPUs > 1` for DataParallel
- Models automatically wrapped in `nn.DataParallel`

**Mixed Precision:**
- Set `--precision=half` for FP16 training

## Key Files to Reference

**Configuration:**
- `codes/option.py`: All command-line arguments and defaults
- `codes/data/__init__.py`: Dataset configuration and paths

**Model Definitions:**
- `codes/model/transenet.py:1-100`: TransENet architecture
- `codes/model/symunet_pretrain.py`: Pre-upsampling SymUNet
- `codes/model/symunet_posttrain.py`: Post-upsampling SymUNet

**Training Infrastructure:**
- `codes/trainer.py:13-50`: Training loop initialization
- `codes/demo_train.py:1-24`: Main training entry point
- `codes/utils.py:64-98`: Checkpoint system

**Testing:**
- `codes/demo_deploy.py:35-150`: Model performance testing
- `codes/metric_scripts/calculate_PSNR_SSIM.py`: Evaluation metrics

## Dataset Downloads

**UCMerced:** [Baidu Drive](https://pan.baidu.com/s/1ijFUcLozP2wiHg14VBFYWw) (password: terr) | [Google Drive](https://drive.google.com/file/d/12pmtffUEAhbEAIn_pit8FxwcdNk4Bgjg/view)

**AID:** [Baidu Drive](https://pan.baidu.com/s/1Cf-J_YdcCB2avPEUZNBoCA) (password: id1n) | [Google Drive](https://drive.google.com/file/d/1d_Wq_U8DW-dOC3etvF4bbbWMOEqtZwF7/view)

**Pre-trained Models:** [Baidu Drive](https://pan.baidu.com/s/1lvAyTagbBf5GWUOcuEkyrQ) (password: w7ct) | [Google Drive](https://drive.google.com/file/d/19nH1Plh2M-Z47iXG0-Ghq-Orh33n787w/view)

## Troubleshooting

**CUDA OOM Errors:**
- Reduce `--batch_size`
- Use SymUNet-Posttrain variant
- Enable `--chop` for memory-efficient inference
- Reduce `--symunet_width`

**Slow Training:**
- Check `--n_threads` for data loading (default: 4)
- Verify GPU utilization with `nvidia-smi`
- Use smaller `--patch_size`
- Try SymUNet-Posttrain for faster iteration

**Import Errors:**
- Ensure conda environment is activated: `conda activate symunet`
- Run from `codes/` directory
- Check `requirements.txt` dependencies

## Reference

**Paper:** Lei, S., Shi, Z., & Mo, W. (2021). "Transformer-based Multi-Stage Enhancement for Remote Sensing Image Super-Resolution." IEEE TGRS.

**Built On:**
- RCAN (Pytorch): https://github.com/yulunzhang/RCAN
- EDSR (Pytorch): https://github.com/sanghyun-son/EDSR-PyTorch
