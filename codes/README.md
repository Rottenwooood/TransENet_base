# SymUNet Engine v2.0 Codebase

This directory contains the source code for SymUNet Engine v2.0, refactored for modularity, scientific consistency, and ease of use.

## Directory Structure
- **`tools/`**: Main entry points for training and testing.
  - `train.py`: Unified training script (YAML-driven).
  - `test.py`: Unified evaluation script (Registry-driven).
  - `run_batch.py`: Batch execution utility.
- **`model/`**: Model definitions.
  - Models are registered via `@ARCH_REGISTRY.register`.
  - To add a new model, create a file here and decorate the class or factory function.
- **`loss/`**: Loss functions.
  - Losses are registered via `@LOSS_REGISTRY.register`.
- **`data/`**: Dataset implementations.
- **`utils/`**: Utility modules.
  - `registry.py`: Model/Loss registration system.
  - `config.py`: Configuration adapter (YAML -> Namespace).
  - `metrics.py`: Standardized PSNR/SSIM implementation.
- **`legacy/`**: Deprecated scripts and original implementations.

## Usage

### Training
Use `codes/tools/train.py` with a YAML configuration file.

```bash
python codes/tools/train.py --config configs/experiments/my_experiment.yaml
```

**Configuration Example (`configs/base.yaml`):**
```yaml
model: SymUNet_Pretrain
epochs: 300
batch_size: 16
scale: 4
use_wandb: true
...
```

### Evaluation
Use `codes/tools/test.py` to evaluate a trained model.

```bash
python codes/tools/test.py --config configs/experiments/my_experiment.yaml \
                           --checkpoint experiments/my_exp/model/model_best.pt \
                           --input_dir datasets/test/LR \
                           --output_dir results/test/SR
```

## Key Changes in v2.0
1.  **Registry Pattern**: Models and losses are decoupled from hardcoded logic.
2.  **Configuration Isolation**: `option.py` dependency removed for new tools. Configs are loaded via `ConfigToArgsAdapter`.
3.  **Scientific Metrics**: Default metrics now use `utils.metrics`, consistent with MATLAB standards (verified against legacy scripts).
4.  **WandB Integration**: Enhanced logging and sweep support.

## Legacy Compatibility
Legacy scripts moved to `codes/legacy/`. They may still function but are not maintained.
