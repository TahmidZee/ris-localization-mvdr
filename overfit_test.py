#!/usr/bin/env python3
"""
Overfit test: Can the model memorize a tiny dataset?
====================================================
This is the DECISIVE architecture diagnostic. We train on ~200 samples for
many epochs and check whether RMSE drops to near-zero.

Result interpretation:
  - φ RMSE < 3° on training data → Architecture CAN learn geometry → problem is optimization/regularization
  - φ RMSE stuck > 10° → Architecture CANNOT represent the input→geometry mapping → need arch changes

Usage:
    python overfit_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ris_pytorch_pipeline.configs import cfg, mdl_cfg, set_seed

# === CRITICAL: Override configs for overfit test ===
# No curriculum delays - pure geometry learning
mdl_cfg.GEOM_ONLY_EPOCHS = 0          # No geometry-only warmup
# IMPORTANT: Never start NMSE ramp in overfit mode (we want a pure-geometry memorization test).
# Using 0.0 ensures the condition `best_aux_phi_rmse < threshold` never triggers (RMSE >= 0).
mdl_cfg.NMSE_RAMP_AUX_PHI_THRESHOLD = 0.0
mdl_cfg.STRUCTURED_COV_WARMUP_EPOCHS = 0      # No NMSE ramp
mdl_cfg.EMA_EVAL_WARMUP_EPOCHS = 9999         # No EMA

# Aggressive learning rates for memorization
# IMPORTANT: Trainer reads `LR_INIT`, not `LR`.
mdl_cfg.LR_INIT = 1e-3                # Higher backbone LR (Trainer uses this)
mdl_cfg.HEAD_LR_MULTIPLIER = 4.0      # Higher head LR
mdl_cfg.BIAS_LR_FINAL_MULTIPLIER = 1.0  # Full LR for bias (no constraint)
mdl_cfg.CLIP_NORM = 10.0              # Relax gradient clipping

# Remove regularization & stochasticity (classic overfit setup)
mdl_cfg.WEIGHT_DECAY = 0.0
mdl_cfg.DROPOUT = 0.0
mdl_cfg.USE_AMP = False

# Full-batch training on the tiny dataset (GPU-cache loader uses drop_last=True)
mdl_cfg.BATCH_SIZE = 200

# CRITICAL: Slot head init that allows backbone gradients to flow
mdl_cfg.SLOT_QUERY_INIT_STD = 1.0     # was 3.0; queries drowned cross-attention signal
# NOTE: _slot_last_linear init is now std=0.05 in model.py (atan activation, was tanh+0.10)

# Keep diversity and sorted losses to help differentiate slots
mdl_cfg.LAM_SLOT_DIVERSITY = 0.5
mdl_cfg.LAM_AUX_SORTED = 2.0
mdl_cfg.LAM_AUX_MASK_BCE = 0.2

# Phase: PURE GEOMETRY test (no NMSE)
# The overfit test isolates geometry learning. NMSE (structural R) adds a second
# gradient path that can fight geometry gradients and cause divergence (e.g., all
# slots converging to +60° because NMSE gradient through steering vectors dominates).
# If geometry works here, we add NMSE back in the full training.
#
# CRITICAL: Disable lam_cov_pred as well (it is an auxiliary NMSE term on R_pred).
# Otherwise the test is not "pure geometry" even if lam_cov=0.
mdl_cfg.LAM_COV_PRED = 0.0
cfg.LAM_COV_PRED = 0.0
cfg.PHASE_LOSS = cfg.PHASE_LOSS or {}
cfg.PHASE_LOSS["joint"] = {
    "lam_cov": 0.0,   # NO NMSE — pure geometry learning test
    "lam_aux": 1.0,   # Geometry loss active
    "lam_subspace_align": 0.0,
    "lam_peak_contrast": 0.0,
    "lam_gap": 0.0,
    "lam_margin": 0.0,
}
cfg.TRAIN_PHASE = "joint"

# Disable early stopping (we WANT to overfit)
mdl_cfg.PATIENCE = 9999

# Use a separate results dir so we don't resume from / overwrite the real training checkpoint
cfg.RESULTS_DIR = "results_overfit_test"
cfg.CKPT_DIR = "results_overfit_test/checkpoints"
os.makedirs(cfg.CKPT_DIR, exist_ok=True)

# Disable auto-resume (start from scratch every time)
cfg.AUTO_RESUME_TRAINING = False

print("=" * 70)
print("OVERFIT TEST: Training on ~200 samples for 200 epochs")
print("=" * 70)
print(f"  LR_INIT backbone={mdl_cfg.LR_INIT}, head={mdl_cfg.LR_INIT * mdl_cfg.HEAD_LR_MULTIPLIER}")
print(f"  BATCH_SIZE={mdl_cfg.BATCH_SIZE} (full-batch), WEIGHT_DECAY={mdl_cfg.WEIGHT_DECAY}, DROPOUT={mdl_cfg.DROPOUT}, AMP={mdl_cfg.USE_AMP}")
print(f"  GEOM_ONLY_EPOCHS={mdl_cfg.GEOM_ONLY_EPOCHS}")
print(f"  NMSE_RAMP_AUX_PHI_THRESHOLD={mdl_cfg.NMSE_RAMP_AUX_PHI_THRESHOLD}")
print(f"  Phase loss: {cfg.PHASE_LOSS['joint']}")
print(f"  CLIP_NORM={mdl_cfg.CLIP_NORM}")
print("=" * 70)
print("Expected: if architecture works, φ RMSE should drop to < 5° within 100 epochs")
print("If φ stays > 15°, the architecture cannot learn the mapping.\n")

set_seed(42)

from ris_pytorch_pipeline.train import Trainer

t = Trainer()
t.fit(
    epochs=200,
    use_shards=True,
    n_train=200,
    n_val=200,
    early_stop_patience=None,
)
