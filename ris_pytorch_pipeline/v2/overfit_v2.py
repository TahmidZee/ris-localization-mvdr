"""
V2 overfit diagnostic.

Goal: verify the covariance model can memorize a fixed tiny subset.
"""

from __future__ import annotations
import argparse

from .config_v2 import v2_mdl
from .dataset_v2 import build_dataloaders_v2
from .model_v2 import CovariancePredictor
from .train_v2 import V2Trainer


def run_overfit_v2(
    n_samples: int = 200,
    epochs: int = 200,
    lr_backbone: float = 1e-3,
    head_lr_mult: float = 4.0,
):
    # Classic overfit setup: high LR, no regularization, deterministic tiny subset.
    v2_mdl.BATCH_SIZE = int(n_samples)
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

    model = CovariancePredictor(dropout=0.0)
    trainer = V2Trainer(model=model, use_amp=False)
    tr_loader, va_loader = build_dataloaders_v2(
        n_train=n_samples,
        n_val=n_samples,
        batch_size=n_samples,
        seed=1337,
        shuffle_train=False,
    )

    history = trainer.fit(
        tr_loader,
        va_loader,
        epochs=epochs,
        train_max_batches=1,
        val_max_batches=1,
        save_best=True,
    )
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
    parser.add_argument("--n", type=int, default=200, help="Subset size (and batch size).")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Backbone learning rate.")
    parser.add_argument("--head-lr-mult", type=float, default=4.0, help="Head LR multiplier.")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_overfit_v2(
        n_samples=args.n,
        epochs=args.epochs,
        lr_backbone=args.lr,
        head_lr_mult=args.head_lr_mult,
    )


if __name__ == "__main__":
    main()
