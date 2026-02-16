"""
V2 trainer: simplified covariance-only training loop.

No curriculum. No slot losses. No permutation matching.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..configs import set_seed
from ..covariance_utils import build_effective_cov_torch
from .config_v2 import v2_cfg, v2_mdl
from .dataset_v2 import build_dataloaders_v2
from .loss_v2 import V2CovarianceLoss
from .model_v2 import CovariancePredictor


class V2Trainer:
    def __init__(
        self,
        model: Optional[CovariancePredictor] = None,
        loss_fn: Optional[V2CovarianceLoss] = None,
        device: Optional[torch.device] = None,
        use_amp: Optional[bool] = None,
        skip_nmse_eff: bool = False,
    ):
        seed = int(getattr(v2_mdl, "SEED", 42))
        set_seed(seed)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model or CovariancePredictor()).to(self.device)
        self.loss_fn = loss_fn or V2CovarianceLoss()
        self.use_amp = bool(v2_mdl.USE_AMP if use_amp is None else use_amp) and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.optimizer = self._build_optimizer()
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(v2_mdl.EPOCHS)),
            eta_min=float(getattr(v2_mdl, "LR_MIN", 1e-6)),
        )

        self.skip_nmse_eff = skip_nmse_eff
        self._global_step = 0

        self.best_val = float("inf")
        Path(v2_cfg.CKPT_DIR).mkdir(parents=True, exist_ok=True)
        Path(v2_cfg.LOGS_DIR).mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self):
        backbone, head = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "factor_head" in name:
                head.append(param)
            else:
                backbone.append(param)

        lr_backbone = float(v2_mdl.LR_BACKBONE)
        lr_head = lr_backbone * float(v2_mdl.HEAD_LR_MULT)
        return AdamW(
            [
                {"params": backbone, "lr": lr_backbone, "name": "backbone"},
                {"params": head, "lr": lr_head, "name": "head"},
            ],
            weight_decay=float(v2_mdl.WEIGHT_DECAY),
        )

    def _unpack_batch(self, batch):
        # CPU-IO path from `ShardNPZDataset`.
        if isinstance(batch, dict):
            y = batch["y"]
            H = batch["H"]
            codes = batch["codes"]
            ptr = batch["ptr"]
            K = batch["K"]
            R_true = batch["R"]
            R_f_true = batch.get("R_f", None)
            snr = batch.get("snr", batch.get("snr_db", None))
            H_taps = batch.get("H_taps", None)
        # TensorDataset path, if used later.
        elif isinstance(batch, (list, tuple)):
            y, H, codes, ptr, K, R_true = batch[:6]
            R_f_true = None
            snr = None
            H_taps = None
            if len(batch) > 6:
                cand = batch[6]
                # If the 7th element looks like per-tone covariance, treat it as R_f_true.
                if torch.is_tensor(cand) and cand.dim() >= 4:
                    R_f_true = cand
                    snr = batch[7] if len(batch) > 7 else None
                    H_taps = batch[8] if len(batch) > 8 else None
                else:
                    # Backward-compatible tuple layout: (.., R_true, snr, H_taps)
                    snr = cand
                    H_taps = batch[7] if len(batch) > 7 else None
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        y = y.to(self.device, non_blocking=True).float()
        H = H.to(self.device, non_blocking=True).float()
        codes = codes.to(self.device, non_blocking=True).float()
        ptr = ptr.to(self.device, non_blocking=True).float()
        K = K.to(self.device, non_blocking=True).long()
        R_true = R_true.to(self.device, non_blocking=True)
        if R_f_true is not None and torch.is_tensor(R_f_true):
            R_f_true = R_f_true.to(self.device, non_blocking=True)
            if R_f_true.numel() == 0:
                R_f_true = None
        if snr is not None:
            snr = snr.to(self.device, non_blocking=True).float()

        H_taps_dev = None
        if isinstance(H_taps, dict) and len(H_taps) > 0:
            H_taps_dev = {}
            for key, value in H_taps.items():
                if torch.is_tensor(value):
                    H_taps_dev[key] = value.to(self.device, non_blocking=True).float()

        return y, H, codes, ptr, K, R_true, R_f_true, snr, H_taps_dev

    def _step(self, batch, train_mode: bool = True):
        y, H, codes, ptr, K, R_true, R_f_true, snr, H_taps = self._unpack_batch(batch)

        with torch.set_grad_enabled(train_mode):
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
            with autocast_ctx:
                out = self.model(y, H, codes, snr_db=snr, H_taps=H_taps)
                R_pred = out["R_pred"]
                if bool(getattr(v2_cfg, "APPLY_EFFECTIVE_COV_IN_LOSS", False)):
                    R_pred = build_effective_cov_torch(
                        R_pred,
                        snr_db=snr,
                        R_samp=None,
                        beta=0.0,
                        diag_load=True,
                        apply_shrink=(snr is not None),
                        target_trace=float(v2_cfg.N),
                    )
                loss, info = self.loss_fn(
                    R_pred,
                    R_true,
                    R_f_pred=out.get("R_f_pred", None),
                    R_f_true=R_f_true,
                    R_f_mean_pred=out.get("R_f_mean_pred", None),
                    K_true=K,
                    ptr_gt=ptr,
                    snr_db=snr,
                )

        if train_mode:
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                clip = float(v2_mdl.CLIP_NORM)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                clip = float(v2_mdl.CLIP_NORM)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
                self.optimizer.step()

        stats = {"loss": float(loss.item())}
        for key, value in info.items():
            stats[key] = float(value)

        # Report physics-aligned effective-cov NMSE as a diagnostic.
        if not self.skip_nmse_eff:
            with torch.no_grad():
                try:
                    R_eff = build_effective_cov_torch(
                        out["R_pred"].detach(),
                        snr_db=snr,
                        R_samp=None,
                        beta=0.0,
                        diag_load=True,
                        apply_shrink=(snr is not None),
                        target_trace=float(v2_cfg.N),
                    )
                    stats["nmse_eff"] = float(self.loss_fn._nmse(R_eff, R_true).mean().item())
                except Exception:
                    pass
        return stats

    @staticmethod
    def _avg_stats(stats_list):
        if not stats_list:
            return {"loss": 0.0}
        keys = set().union(*[s.keys() for s in stats_list])
        out = {}
        for key in keys:
            vals = [float(s[key]) for s in stats_list if key in s]
            if vals:
                out[key] = float(sum(vals) / len(vals))
        return out

    def train_one_epoch(self, train_loader, max_batches: Optional[int] = None):
        self.model.train()
        running = []
        for bi, batch in enumerate(train_loader):
            if max_batches is not None and bi >= max_batches:
                break
            is_first = (self._global_step == 0)
            if is_first:
                print("[V2] first batch loaded → forward+backward ...", flush=True)
            t0 = time.time()
            running.append(self._step(batch, train_mode=True))
            self._global_step += 1
            if is_first:
                dt = time.time() - t0
                print(f"[V2] first step done in {dt:.2f}s", flush=True)
        return self._avg_stats(running)

    @torch.no_grad()
    def validate(self, val_loader, max_batches: Optional[int] = None):
        self.model.eval()
        running = []
        for bi, batch in enumerate(val_loader):
            if max_batches is not None and bi >= max_batches:
                break
            running.append(self._step(batch, train_mode=False))
        return self._avg_stats(running)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: Optional[int] = None,
        train_max_batches: Optional[int] = None,
        val_max_batches: Optional[int] = None,
        save_best: bool = True,
    ):
        n_epochs = int(epochs if epochs is not None else v2_mdl.EPOCHS)
        history = []
        for epoch in range(1, n_epochs + 1):
            tr = self.train_one_epoch(train_loader, max_batches=train_max_batches)
            va = self.validate(val_loader, max_batches=val_max_batches)
            self.scheduler.step()

            lrs = [group["lr"] for group in self.optimizer.param_groups]
            print(
                f"Epoch {epoch:03d}/{n_epochs:03d} "
                f"train_loss={tr.get('loss', 0.0):.6f} "
                f"val_loss={va.get('loss', 0.0):.6f} "
                f"nmse(train/val)={tr.get('nmse', 0.0):.6f}/{va.get('nmse', 0.0):.6f} "
                f"lr={lrs}",
                flush=True,
            )

            history.append({"epoch": epoch, "train": tr, "val": va, "lr": lrs})
            if save_best and float(va.get("loss", 1e9)) < self.best_val:
                self.best_val = float(va["loss"])
                self.save_checkpoint(Path(v2_cfg.CKPT_DIR) / "best_v2.pt", epoch=epoch)

        return history

    def save_checkpoint(self, path: Path, epoch: int):
        payload = {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val": float(self.best_val),
            "v2_cfg": vars(v2_cfg),
            "v2_mdl": vars(v2_mdl),
        }
        torch.save(payload, str(path))


def run_v2_training(
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    tr_loader, va_loader = build_dataloaders_v2(
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        seed=1337,
        shuffle_train=True,
    )
    trainer = V2Trainer()
    return trainer.fit(tr_loader, va_loader, epochs=epochs)
