# Enhanced Training Features

This document describes the enhanced training features added to the SymUNet training pipeline.

## 🆕 New Features

### 1. WandB Integration
- **Real-time experiment tracking**
- **Automatic logging of training metrics**
- **Visualization of training progress**
- **Model artifact saving**

### 2. AdamW Optimizer
- **Better generalization than Adam**
- **Decoupled weight decay**
- **Improved training stability**

### 3. Cosine Annealing Learning Rate Scheduling
- **Smooth learning rate decay**
- **Better convergence properties**
- **Configurable minimum learning rate**

### 4. Step-based Checkpoint Saving
- **Save checkpoints every N steps**
- **Better training recovery**
- **Fine-grained experiment tracking**

## 📝 Usage Examples

### Basic Enhanced Training

```bash
python train_enhanced.py \
    --model SYMUNET_PRETRAIN \
    --dataset=UCMerced \
    --scale 4 \
    --epochs 300 \
    --batch_size 4 \
    --optimizer ADAMW \
    --scheduler cosine \
    --lr 2e-4 \
    --cosine_t_max 300 \
    --cosine_eta_min 5e-5 \
    --loss "1*L1+0.005*FFT" \
    --save enhanced_training
```

### Training with WandB

```bash
python train_enhanced.py \
    --model SYMUNET_PRETRAIN \
    --dataset=UCMerced \
    --scale 4 \
    --use_wandb \
    --wandb_project "My-SR-Project" \
    --wandb_name "symunet_pretrain_v1" \
    --epochs 300 \
    --batch_size 4 \
    --optimizer ADAMW \
    --scheduler cosine \
    --lr 2e-4 \
    --save enhanced_training
```

### Step-based Checkpoint Saving

```bash
python train_enhanced.py \
    --model SYMUNET_PRETRAIN \
    --dataset=UCMerced \
    --scale 4 \
    --save_every_n_steps 50 \
    --epochs 300 \
    --batch_size 4 \
    --optimizer ADAMW \
    --scheduler cosine \
    --lr 2e-4 \
    --save enhanced_training
```

### Resume Training with New Settings

```bash
python train_enhanced.py \
    --resume 1 \
    --save enhanced_training \
    --optimizer ADAMW \
    --scheduler cosine \
    --lr 1e-4
```

## 🔧 Parameter Reference

### WandB Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_wandb` | Enable WandB logging | False |
| `--wandb_project` | WandB project name | "SymUNet-SR" |
| `--wandb_entity` | WandB entity name | None |
| `--wandb_name` | Experiment name | Auto-generated |

### Enhanced Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--scheduler` | Learning rate scheduler | "step" |
| `--cosine_t_max` | Max steps for cosine annealing | 300 |
| `--cosine_eta_min` | Minimum learning rate | 5e-5 |
| `--save_every_n_steps` | Save checkpoint every N steps | 50 |

### Optimizer Parameters

| Optimizer | Parameters | Description |
|-----------|------------|-------------|
| ADAMW | `lr`, `weight_decay`, `betas`, `eps` | AdamW optimizer |
| ADAM | `lr`, `weight_decay`, `betas`, `eps` | Adam optimizer |
| SGD | `lr`, `momentum`, `weight_decay` | SGD optimizer |
| RMSprop | `lr`, `weight_decay`, `eps` | RMSprop optimizer |

## 📊 WandB Dashboard

When WandB is enabled, you can monitor:

### Training Metrics
- **Loss curves** (L1, FFT, Total)
- **Learning rate** progression
- **Training time** per epoch
- **Gradient norms**

### Validation Metrics
- **PSNR** values
- **SSIM** values (if enabled)
- **Best model** tracking

### Model Information
- **Parameter count**
- **Architecture summary**
- **Training configuration**

### Sample Images
- **Input LR images**
- **Generated SR images**
- **Ground truth HR images**
- **Side-by-side comparisons**

## 🎯 Recommended Settings

### For Remote Sensing SR (Your Use Case)

```bash
python train_enhanced.py \
    --model SYMUNET_PRETRAIN \
    --dataset=UCMerced \
    --scale 4 \
    --epochs 300 \
    --batch_size 4 \
    --optimizer ADAMW \
    --scheduler cosine \
    --lr 2e-4 \
    --cosine_t_max 300 \
    --cosine_eta_min 5e-5 \
    --loss "1*L1+0.005*FFT" \
    --symunet_pretrain_width 48 \
    --symunet_pretrain_enc_blk_nums 4,6,6 \
    --symunet_pretrain_dec_blk_nums 6,6,4 \
    --symunet_pretrain_restormer_heads 1,2,4 \
    --symunet_pretrain_restormer_middle_heads 8 \
    --save_every_n_steps 50 \
    --use_wandb \
    --wandb_project "SymUNet-SR" \
    --wandb_name "ucmerced_x4_pretrain" \
    --save symunet_enhanced_x4
```

### Key Advantages

1. **AdamW**: Better generalization for remote sensing images
2. **Cosine Annealing**: Smoother convergence than step decay
3. **WandB**: Complete experiment tracking and visualization
4. **Step Checkpoints**: Better recovery and fine-grained control

## 🚨 Important Notes

### WandB Setup
1. Install WandB: `pip install wandb`
2. Login to WandB: `wandb login`
3. Or use API key in environment: `export WANDB_API_KEY=your_key`

### Memory Considerations
- With `batch_size=4` and `width=48`, you should have sufficient GPU memory
- If OOM occurs, reduce `batch_size` or `width`

### Checkpoint Management
- Regular epoch-based checkpoints are still saved
- Additional step-based checkpoints are saved every N steps
- Step checkpoints contain: model, optimizer, scheduler states

### Resume Training
- Resume works with all new features
- Learning rate and scheduler states are preserved
- WandB run will continue if `--use_wandb` is enabled

## 📈 Expected Performance

### With AdamW + Cosine Annealing
- **Faster convergence** in initial epochs
- **Smoother loss curves**
- **Better final performance**
- **More stable training**

### With WandB Monitoring
- **Complete experiment tracking**
- **Easy hyperparameter tuning**
- **Performance analysis**
- **Reproducible experiments**

## 🔍 Troubleshooting

### Common Issues

1. **WandB login fails**:
   ```bash
   wandb login --relogin
   ```

2. **CUDA OOM**:
   ```bash
   # Reduce batch size
   --batch_size 2

   # Or reduce model width
   --symunet_pretrain_width 32
   ```

3. **Slow training**:
   ```bash
   # Increase batch size if memory allows
   --batch_size 8

   # Use mixed precision
   --precision half
   ```

4. **Scheduler issues**:
   ```bash
   # For cosine annealing, ensure total steps match
   --cosine_t_max 300  # Should be <= epochs
   ```

## 📚 References

- [WandB Documentation](https://docs.wandb.ai/)
- [AdamW Paper](https://arxiv.org/abs/1711.05101)
- [Cosine Annealing](https://arxiv.org/abs/1608.03983)
- [PyTorch Schedulers](https://pytorch.org/docs/stable/optim.html)

Happy training! 🚀