# QA Report: SymUNet Engine v2.0 Refactoring

## 1. Summary
This report summarizes the verification and testing process for the SymUNet Engine v2.0 refactoring. The goal was to modernize the codebase, introduce a Registry pattern for models and losses, implement configuration isolation via adapters, and ensure scientific consistency with legacy implementations.

**Overall Status:** ✅ **PASSED**

## 2. Test Execution Details

### 2.1 Unit Tests: Configuration Adapter
- **Test File:** `codes/tests/test_config_adapter.py`
- **Purpose:** Verify that nested dictionaries from YAML are correctly converted to `argparse.Namespace` objects.
- **Result:** PASSED (3 tests)
- **Key Validation:**
    - Recursive conversion of nested dictionaries.
    - Default value application.

### 2.2 Integration Tests: Training Components
- **Test File:** `codes/tests/test_train_integration.py`
- **Purpose:** Verify that `codes/train.py` correctly integrates with `ARCH_REGISTRY` and `LOSS_REGISTRY`.
- **Result:** PASSED (2 tests)
- **Key Validation:**
    - `ARCH_REGISTRY` correctly builds models.
    - `LOSS_REGISTRY` correctly builds loss functions (including legacy fallbacks).
    - Early stopping hook logic verification.

### 2.3 Scientific Consistency Tests
- **Test File:** `codes/tests/test_scientific_consistency.py`
- **Purpose:** Ensure that new metric implementations (`codes/utils/metrics.py`) produce mathematically identical results to legacy scripts (`codes/legacy/metric_scripts`).
- **Result:** PASSED (2 tests)
- **Key Validation:**
    - PSNR calculation on Y-channel matches legacy MATLAB-style implementation.
    - SSIM calculation matches legacy implementation within tolerance (`1e-4`).

### 2.4 End-to-End Smoke Test
- **Test File:** `codes/tests/test_e2e_train.py`
- **Purpose:** Verify the entire training pipeline from configuration loading to model saving.
- **Result:** PASSED
- **Key Validation:**
    - Parsing of YAML configuration.
    - Data loading (UCMerced format).
    - Model initialization via Registry (fixed `make_model` factory registration).
    - Training loop execution for 1 epoch.
    - Checkpoint saving (verified in experiment subdirectory).
    - No crashes via `subprocess` isolation.

## 3. Known Issues & Resolutions during QA
1.  **Registry Import Order:** Models were not being registered because `codes/model/__init__.py` did not import submodule files.
    - *Fix:* Added automatic module discovery in `codes/model/__init__.py`.
    - *Verified:* `SymUNet_Pretrain` is now correctly registered and built.
2.  **Model Factory Registration:** `SymUNet_Pretrain` class does not accept `args` in `__init__`, causing `TypeError` when Registry called it.
    - *Fix:* Moved `@register` decorator to `make_model` factory function which accepts `args`.
3.  **WandB NoneType Error:** `trainer.terminate()` tried to call `finish()` on `None` logger when WandB was disabled.
    - *Fix:* Added `if self.wandb_logger is not None` guard.
4.  **Output Directory Logic:** Legacy `checkpoint` class ignored configured `dir_out` and used hardcoded `../experiment/`.
    - *Fix:* Updated `codes/utils/common.py` to respect `args.dir_out` if available.
    - *Verified:* E2E test confirms logs are written to the configured output directory.

## 4. Conclusion
The refactoring is complete and verified. The engine is now modular, registry-driven, and maintains scientific accuracy.
