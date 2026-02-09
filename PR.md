# PR: SymUNet Engine v2.0 重构 - 配置解耦、科学性修正与自动化 HPO

**Reviewers**: @LiXiaochen, @Team
**Status**: Ready for Merge
**Type**: Refactoring & Feature
**Impact**: High (Core Training Engine Update)

---

## 1. 摘要 (Executive Summary)

晓琛，针对你之前指出的 **`option.py` 参数解析脆弱** 以及 **`batch_train.py` 隐式并行逻辑导致模型配置错乱** 的严重问题，我对整个工程进行了深度重构（Engine v2.0）。

本次重构的核心哲学是 **"Explicit over Implicit" (显式优于隐式)** 和 **"Smart Generator, Dumb Runner" (智能生成，傻瓜执行)**。

我们彻底废弃了在训练脚本运行时动态推断参数组合的危险逻辑，转而采用 **"配置清单 (Manifest)"** 驱动的确定性执行流。同时，为了满足 CVPR/ICCV 等顶级会议的发表要求，我对评估指标计算进行了严格的科学性修正（对齐 MATLAB 标准），并引入了 WandB Sweeps 进行自动化的架构搜索。

**核心改进点：**

1. **彻底解决配置混乱**：移除 `batch_train.py` 所有逻辑，改为执行静态 JSON 清单。
2. **科学严谨性**：PSNR/SSIM 指标计算从 RGB 修正为 YCbCr-Y 通道，并实现了标准的 Border Shaving。
3. **模块化解耦**：引入 `Registry` 模式管理模型与 Loss，解耦了 `trainer.py` 对全局 `args` 的强依赖。
4. **自动化 HPO**：原生支持 WandB Sweeps，支持大规模架构搜索与早停（Early Stopping）。

---

## 2. 目录架构变更说明 (Directory Structure)

为了实现模块化，我重新梳理了 `codes/` 下的目录结构，将原作者的遗留代码与新版引擎进行了物理隔离。

```text
codes/
├── tools/                  # [NEW] 新版单一入口工具集
│   ├── train.py            # 支持 YAML/WandB 的统一训练入口
│   ├── test.py             # 基于 Registry 的标准化测试脚本
│   ├── run_batch.py        # "傻瓜式" 批量执行器 (替代原 batch_train.py)
│   ├── generate_configs.py # 显式配置生成器 (解决 Li's Problem 的核心)
│   └── generate_manifest.py# YAML -> JSON 清单编译工具
├── configs/                # [NEW] 实验配置文件仓库
│   ├── base.yaml           # 基础超参定义 (从 option.py 提取)
│   ├── sweeps/             # WandB 搜索空间定义
│   └── experiments/        # 具体实验配置
├── utils/                  # [NEW] 通用工具库
│   ├── registry.py         # 注册机 (Model/Loss)
│   ├── config.py           # 配置适配器 (YAML -> Argparse Namespace)
│   └── metrics.py          # 科学级指标计算 (Y-channel PSNR)
├── model/                  # 模型定义 (已通过 Registry 解耦)
├── loss/                   # 损失函数 (已通过 Builder 解耦)
├── tests/                  # [NEW] 完整的单元测试套件
└── legacy/                 # [MOVED] 原作者的旧脚本及废弃的 batch_deploy

```

---

## 3. 核心重构详解：如何解决 "Li's Problem"

### 3.1 问题回顾

之前的 `batch_train.py` 试图通过判断参数列表长度（例如 `enc_blk_nums=[2,2], [4,4]`）来自动生成实验组合。这种隐式逻辑极其脆弱，当需要同时调整宽度（Width）和深度（Depth）时，很容易因为列表长度对齐问题导致训练出非预期的模型架构。

### 3.2 解决方案：Manifest-Based Execution

现在的流程变为三步走：

1. **Generate (生成)**: 使用 Python 脚本或 YAML 显式定义搜索空间，生成一个静态的 `experiments_manifest.json`。此时你可以人工检查 JSON，确保每一个实验的参数都是你想要的。
2. **Verify (验证)**: 此时代码还没跑，你可以确信配置是无误的。
3. **Execute (执行)**: `run_batch.py` 读取 JSON，仅仅是简单地串行执行命令，不做任何逻辑推理。

### 3.3 上手指南

#### 方式一：使用 Python 脚本生成复杂组合 (推荐)

查看 `codes/tools/generate_configs.py`。在这里，你可以用 Python 代码显式定义哪些参数是成对出现的，避免了笛卡尔积的错误组合。

```python
# codes/tools/generate_configs.py 示例
def generate_experiments():
    # 显式定义合法的架构对 (Encoder - Decoder)，不再依赖列表索引对齐
    valid_archs = [
        {"enc": "2,2,2", "dec": "2,2,2", "width": 32}, # Small
        {"enc": "4,6,6", "dec": "6,6,4", "width": 64}, # Large
    ]
    
    experiments = []
    for arch in valid_archs:
        config = base_config.copy()
        config["symunet_pretrain_width"] = arch["width"]
        config["symunet_pretrain_enc_blk_nums"] = arch["enc"]
        # ... 生成配置字典 ...
        experiments.append(config)
    return experiments

```

**运行命令：**

```bash
# 1. 生成清单
python codes/tools/generate_configs.py
# 输出: Generated 6 experiments in experiments_manifest.json

# 2. 执行训练
python codes/tools/run_batch.py --manifest experiments_manifest.json

```

#### 方式二：使用 YAML 定义网格搜索

如果你确实需要做简单的网格搜索，可以使用 `configs/` 下的 YAML 配置配合 `generate_manifest.py`。

---

## 4. 科学性修正：PSNR/SSIM 评估标准

这是论文发表的关键。之前的 `batch_deploy_psnr.py` 直接计算 RGB 通道的 PSNR，这在 SR 领域是不规范的（数值通常偏低且不稳定）。

我在 `codes/utils/metrics.py` 中重写了评估逻辑，并经由 `codes/tests/test_scientific_consistency.py` 验证通过。

### 4.1 修正细节

1. **颜色空间转换**：实现了标准的 `RGB -> YCbCr` 转换。
* **关键点**：OpenCV 的 `cvtColor` 和 MATLAB 的 `rgb2ycbcr` 系数有细微差别。为了保证与学术界 Benchmark (如 SwinIR, RCAN) 对齐，我手动实现了基于 MATLAB 系数的转换公式：
```python
Y = 65.481 * R + 128.553 * G + 24.966 * B + 16.0  # (Input 0-255)

```


2. **Y 通道评估**：仅在 Y (Luminance) 通道上计算指标。
3. **Border Shaving**：根据缩放倍数（Scale），在计算指标前必须裁掉图像边缘的 `Scale` 个像素。之前的代码虽然有变量，但在某些路径下未生效。现在这是强制的。

### 4.2 验证结果

在 `tests/test_metrics.py` 中，我构建了特定颜色的测试用例（如纯蓝 `(255,0,0)`），验证转换后的 Y 值严格等于 41（MATLAB 标准值），确保了指标的权威性。

---

## 5. WandB HPO 大规模搜索指南

为了解决“手动调参效率低”的问题，我集成了 WandB Sweeps。这允许我们定义超参范围，让 WandB 的贝叶斯优化算法帮我们寻找最优解。

### 5.1 配置文件

请查看 `configs/sweeps/symunet_arch_search.yaml`。

```yaml
program: codes/tools/train.py
method: bayes  # 贝叶斯优化
metric:
  name: val/psnr
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  # 架构搜索：这里使用了 Categorical 映射技巧
  # 因为 WandB 传列表参数很麻烦，我们在代码里定义了 ARCH_CONFIGS 字典
  arch_id:
    values: [0, 1, 2, 3] 

```

### 5.2 代码适配 (`train.py`)

在 `codes/tools/train.py` 中，我注入了 `update_args_with_wandb` 钩子。当它检测到处于 Sweep 模式时，会优先使用 WandB 下发的参数覆盖本地配置。

### 5.3 如何启动搜索

```bash
# 1. 初始化 Sweep (在 WandB 服务器注册)
wandb sweep configs/sweeps/symunet_arch_search.yaml
# 输出: wandb: Created sweep with ID: xxxxxxxx

# 2. 启动 Agent (可以在多台机器上运行此命令，自动并行)
wandb agent <你的Entity>/SymUNet-SR/xxxxxxxx

```

---

## 6. 代码质量与测试 (QA)

我不希望像以前一样“跑起来全靠运气”。这次重构引入了完整的单元测试套件（位于 `codes/tests/`）。

* **`test_config_adapter.py`**: 验证 YAML 嵌套字典能不能正确转成 `argparse.Namespace`，确保兼容旧代码的 `args.param` 访问方式。
* **`test_registry.py`**: 验证新的 `ARCH_REGISTRY` 能否正确根据字符串名称构建模型。
* **`test_loss_builder.py`**: 验证复合损失函数字符串（如 `1*L1+0.05*StableFFT`）能否被正确解析和构建。
* **`test_e2e_train.py`**: **最重要的冒烟测试**。它会在临时目录构建一个微型数据集，跑 1 个 epoch 的训练。
* 每次修改代码后，请务必运行：`pytest codes/tests/test_e2e_train.py`，确保没有低级语法错误。

---

## 7. 其他关键改动

1. **Loss 模块化**：
* 新增 `codes/loss/builder.py`。以前修改 Loss 组合需要改 `__init__.py` 的硬编码逻辑。现在只需在 Config 字符串里写 `Weight*LossName`，Builder 会自动从 Registry 中查找并组合。
* 新增 `StableFFT` Loss：为了解决 FFT Loss 在训练初期不稳定的问题，我增加了一个 `StableFFTLoss`，默认开启了 `ortho` 归一化并忽略 DC 分量（直流分量）。


2. **通用工具类**：
* `codes/utils/common.py` 中的 `checkpoint` 类修复了一个 Bug：它以前硬编码了输出路径为 `../experiment`，现在它会优先尊重要求配置中的 `dir_out`。


3. **Requirements**：
* 新增 `pyproject.toml`，规范了依赖版本（torch, numpy, wandb 等）。

---

## 总结

这次重构没有改动原作者的核心算法实现（SRCNN, VDSR 等），而是重塑了**实验的外围基础设施**。

* **对李晓琛**：直接使用 `codes/tools/generate_configs.py` 生成你的实验清单，然后跑 `run_batch.py`。再也不用担心参数错乱了。
* **对论文投稿**：现在的评估指标（`metrics.py`）是经得起推敲的。
* **对未来扩展**：新增模型只需在 `model/` 下新建文件并加上 `@ARCH_REGISTRY.register` 即可，无需修改其他任何文件。

请 Review 代码，若无异议，建议合并到 `main` 分支并作为后续实验的基准（Baseline）。