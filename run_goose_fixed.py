#!/usr/bin/env python3
"""
Hardened training script for goose that explicitly sets and verifies loss weights.
This bypasses any config file issues and ensures the correct values are active.
"""
import random
import numpy as np
import torch
from pathlib import Path
from ris_pytorch_pipeline.configs import cfg, mdl_cfg, set_seed
from ris_pytorch_pipeline.train import Trainer

# === Reproducibility ===
set_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# === Training settings ===
mdl_cfg.SEED = 42
mdl_cfg.TRAIN_USE_GPU_CACHE = True  # Use GPU cache for full run
mdl_cfg.USE_EMA = True
mdl_cfg.USE_SWA = True
mdl_cfg.BATCH_SIZE = 64

cfg.AUTO_RESUME_TRAINING = False
cfg.AUTO_RESUME_WEIGHTS_ONLY = False
cfg.CKPT_DIR = "results_M64_N256_L64/checkpoints"
cfg.LOGS_DIR = "results_M64_N256_L64/logs"
Path(cfg.CKPT_DIR).mkdir(parents=True, exist_ok=True)
Path(cfg.LOGS_DIR).mkdir(parents=True, exist_ok=True)

# === CRITICAL: Force correct loss weights in config BEFORE Trainer init ===
# These should match what's in configs.py, but we force them explicitly here
# to ensure they're active regardless of any config file issues.
cfg.PHASE_LOSS["joint"]["lam_aux"] = 1.0
mdl_cfg.LAM_AUX_SORTED = 1.5
mdl_cfg.LAM_AUX_MASK_BCE = 0.2

print("=" * 80, flush=True)
print("🔧 HARDENED CONFIG OVERRIDES (goose-safe):", flush=True)
print(f"  cfg.PHASE_LOSS['joint']['lam_aux'] = {cfg.PHASE_LOSS['joint']['lam_aux']}", flush=True)
print(f"  mdl_cfg.LAM_AUX_SORTED = {mdl_cfg.LAM_AUX_SORTED}", flush=True)
print(f"  mdl_cfg.LAM_AUX_MASK_BCE = {mdl_cfg.LAM_AUX_MASK_BCE}", flush=True)
print("=" * 80, flush=True)

# === Initialize Trainer ===
t = Trainer(from_hpo=False)

# === VERIFY loss weights are correct AFTER Trainer init ===
print("\n" + "=" * 80, flush=True)
print("✅ FINAL VERIFICATION (after Trainer init):", flush=True)
print(f"  t.loss_fn.lam_aux = {t.loss_fn.lam_aux:.3f} (expect 1.0)", flush=True)
print(f"  t.loss_fn.lam_aux_sorted = {t.loss_fn.lam_aux_sorted:.3f} (expect 1.5)", flush=True)
lam_mask_bce = float(getattr(mdl_cfg, "LAM_AUX_MASK_BCE", 0.0))
print(f"  mdl_cfg.LAM_AUX_MASK_BCE = {lam_mask_bce:.3f} (expect 0.2)", flush=True)

# === CRITICAL: Force them again AFTER Trainer init (defense in depth) ===
if abs(t.loss_fn.lam_aux - 1.0) > 0.01:
    print(f"⚠️  WARNING: lam_aux was {t.loss_fn.lam_aux}, forcing to 1.0", flush=True)
    t.loss_fn.lam_aux = 1.0

if abs(t.loss_fn.lam_aux_sorted - 1.5) > 0.01:
    print(f"⚠️  WARNING: lam_aux_sorted was {t.loss_fn.lam_aux_sorted}, forcing to 1.5", flush=True)
    t.loss_fn.lam_aux_sorted = 1.5

if abs(lam_mask_bce - 0.2) > 0.01:
    print(f"⚠️  WARNING: LAM_AUX_MASK_BCE was {lam_mask_bce}, forcing to 0.2", flush=True)
    mdl_cfg.LAM_AUX_MASK_BCE = 0.2

print("=" * 80 + "\n", flush=True)

# === Run training ===
print("🚀 Starting training with verified loss weights...\n", flush=True)
t.fit(
    epochs=60,
    use_shards=True,
    n_train=100000,
    n_val=10000,
    early_stop_patience=None,  # Use mdl_cfg.PATIENCE default (20)
)
