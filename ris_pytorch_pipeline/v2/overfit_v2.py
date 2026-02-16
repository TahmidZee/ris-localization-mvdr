"""
V2 overfit diagnostic.

Goal: verify the covariance model can memorize a fixed tiny subset.
"""

from __future__ import annotations
import argparse

from .config_v2 import v2_cfg, v2_mdl
from .dataset_v2 import build_dataloaders_v2
from .model_v2 import CovariancePredictor
from .train_v2 import V2Trainer


def _estimate_sample_bytes() -> int:
    """
    Rough per-sample host-memory footprint for wideband overfit batches.
    Uses float32 RI tensors; real runtime memory is higher due to activations.
    """
    L = int(getattr(v2_cfg, "L", 16))
    F = max(1, int(getattr(v2_cfg, "F_SUBCARRIERS", 1)))
    M = int(getattr(v2_cfg, "M", 16))
    N = int(getattr(v2_cfg, "N", 144))
    P = max(1, int(getattr(v2_cfg, "D_TAPS", 8)))
    f32 = 4

    bytes_per = 0
    bytes_per += L * F * M * 2 * f32      # y
    bytes_per += L * F * M * 2 * f32      # H
    bytes_per += L * N * 2 * f32          # codes
    bytes_per += N * N * 2 * f32          # R
    bytes_per += F * N * N * 2 * f32      # R_f
    bytes_per += P * M * N * 2 * f32      # H_taps_ri
    return int(max(1, bytes_per))


def _suggest_overfit_batch(n_samples: int) -> tuple[int, float]:
    """
    Suggest a safer overfit batch for wideband settings.
    Keep raw host batch modest and account for N^2*F scaling in covariance heads.
    """
    bytes_per = _estimate_sample_bytes()
    target_raw_batch_bytes = int(128 * 1024 * 1024)
    cap = max(1, target_raw_batch_bytes // max(1, bytes_per))
    N = int(getattr(v2_cfg, "N", 144))
    F = max(1, int(getattr(v2_cfg, "F_SUBCARRIERS", 1)))
    cov_complexity = N * N * F
    hard_cap = 8 if cov_complexity >= 1_000_000 else 16
    cap = min(cap, hard_cap, max(1, int(n_samples)))
    sample_mib = float(bytes_per) / (1024.0 * 1024.0)
    return int(max(1, cap)), sample_mib


def run_overfit_v2(
    n_samples: int = 200,
    epochs: int = 200,
    lr_backbone: float = 1e-3,
    head_lr_mult: float = 4.0,
    batch_size: int | None = None,
):
    # Classic overfit setup: high LR, no regularization, deterministic tiny subset.
    n_samples = max(1, int(n_samples))
    auto_bs, sample_mib = _suggest_overfit_batch(n_samples)
    eff_bs = int(batch_size) if batch_size is not None else auto_bs
    eff_bs = max(1, min(eff_bs, n_samples))

    v2_mdl.BATCH_SIZE = int(eff_bs)
    v2_mdl.EPOCHS = int(epochs)
    v2_mdl.LR_BACKBONE = float(lr_backbone)
    v2_mdl.HEAD_LR_MULT = float(head_lr_mult)
    v2_mdl.WEIGHT_DECAY = 0.0
    v2_mdl.USE_AMP = False
    v2_mdl.LAM_COV_MAIN = 1.0
    v2_mdl.LAM_COV_F = 1.0
    v2_mdl.LAM_COV_CONSIST = 0.05
    v2_mdl.LAM_SUBSPACE = 0.0
    v2_mdl.LAM_PEAK = 0.0

    steps_per_epoch = (n_samples + eff_bs - 1) // eff_bs
    raw_batch_mib = sample_mib * float(eff_bs)
    print(
        f"[OVERFIT V2] n={n_samples} batch={eff_bs} steps/epoch={steps_per_epoch} "
        f"raw_batch≈{raw_batch_mib:.1f} MiB (sample≈{sample_mib:.2f} MiB)",
        flush=True,
    )
    if eff_bs < n_samples:
        print(
            "[OVERFIT V2] Using mini-batch overfit to avoid OOM from full-batch wideband tensors.",
            flush=True,
        )

    model = CovariancePredictor(dropout=0.0)
    trainer = V2Trainer(model=model, use_amp=False)
    prev_pin_memory = bool(getattr(v2_cfg, "PIN_MEMORY", True))
    v2_cfg.PIN_MEMORY = False
    tr_loader, va_loader = build_dataloaders_v2(
        n_train=n_samples,
        n_val=n_samples,
        batch_size=eff_bs,
        seed=1337,
        shuffle_train=False,
    )

    try:
        history = trainer.fit(
            tr_loader,
            va_loader,
            epochs=epochs,
            train_max_batches=(1 if eff_bs == n_samples else None),
            val_max_batches=(1 if eff_bs == n_samples else None),
            save_best=False,
        )
    finally:
        v2_cfg.PIN_MEMORY = prev_pin_memory
    if history:
        last = history[-1]
        print(
            f"[OVERFIT V2] final train_nmse={last['train'].get('nmse', -1):.6f} "
            f"val_nmse={last['val'].get('nmse', -1):.6f}",
            flush=True,
        )
    return history


def _parse_args():
    parser = argparse.ArgumentParser(description="Run V2 covariance overfit diagnostic.")
    parser.add_argument("--n", type=int, default=200, help="Subset size.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Backbone learning rate.")
    parser.add_argument("--head-lr-mult", type=float, default=4.0, help="Head LR multiplier.")
    parser.add_argument("--batch", type=int, default=None, help="Overfit batch size override (default: auto-safe).")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_overfit_v2(
        n_samples=args.n,
        epochs=args.epochs,
        lr_backbone=args.lr,
        head_lr_mult=args.head_lr_mult,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
