"""
V2 entrypoint for check/train/overfit flows.
"""

from __future__ import annotations
import argparse
from pprint import pformat

from .config_v2 import v2_cfg, v2_mdl
from .dataset_v2 import resolve_shards_train_val
from .loss_v2 import V2CovarianceLoss
from .model_v2 import CovariancePredictor
from .overfit_v2 import run_overfit_v2
from .train_v2 import run_v2_training


def run_check():
    model = CovariancePredictor()
    loss_fn = V2CovarianceLoss()
    print("[V2 CHECK] config loaded", flush=True)
    print("[V2 CHECK] v2_cfg:", flush=True)
    print(pformat(vars(v2_cfg)), flush=True)
    print("[V2 CHECK] v2_mdl:", flush=True)
    print(pformat(vars(v2_mdl)), flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[V2 CHECK] model params={n_params}", flush=True)
    print(f"[V2 CHECK] loss={loss_fn.__class__.__name__}", flush=True)
    print(
        f"[V2 CHECK] plan docs: primary={v2_cfg.PRIMARY_PLAN_DOC}, execution={v2_cfg.V2_EXEC_PLAN_DOC}",
        flush=True,
    )
    try:
        tr, va, is_wb = resolve_shards_train_val()
        print(
            f"[V2 CHECK] data resolved: train={tr} val={va} mode={'wideband' if is_wb else 'narrowband-fallback'}",
            flush=True,
        )
    except Exception as exc:
        print(f"[V2 CHECK] data resolution failed: {exc}", flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="V2 physics-first covariance pipeline runner.")
    parser.add_argument("--check", action="store_true", help="Print v2 config/model summary and exit.")
    parser.add_argument("--check-data", action="store_true", help="Resolve configured data paths and exit.")
    parser.add_argument("--overfit", action="store_true", help="Run overfit diagnostic.")
    parser.add_argument("--train", action="store_true", help="Run regular train/val workflow.")
    parser.add_argument("--n-train", type=int, default=None, help="Optional train subset cap.")
    parser.add_argument("--n-val", type=int, default=None, help="Optional val subset cap.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--overfit-n", type=int, default=200, help="Overfit subset size.")
    parser.add_argument("--overfit-epochs", type=int, default=200, help="Overfit epochs.")
    parser.add_argument("--overfit-lr", type=float, default=1e-3, help="Overfit backbone LR.")
    parser.add_argument("--overfit-head-lr-mult", type=float, default=4.0, help="Overfit head LR multiplier.")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.check:
        run_check()
        return
    if args.check_data:
        _, _, _ = resolve_shards_train_val()
        print("[V2 CHECK DATA] dataset paths resolved successfully", flush=True)
        return

    if args.overfit:
        run_overfit_v2(
            n_samples=args.overfit_n,
            epochs=args.overfit_epochs,
            lr_backbone=args.overfit_lr,
            head_lr_mult=args.overfit_head_lr_mult,
        )
        return

    # Default mode: train, unless user explicitly requested a different one.
    if args.train or (not args.check and not args.overfit):
        run_v2_training(
            n_train=args.n_train,
            n_val=args.n_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
