# hpo.py
import os, time, math
from pathlib import Path
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
import sys
import logging

from .configs import cfg, mdl_cfg, set_seed
from .train import Trainer

def _save_hpo_winner(trial: optuna.trial.FrozenTrial):
    Path(cfg.HPO_DIR).mkdir(parents=True, exist_ok=True)
    payload = {
        "value": float(trial.value),
        "number": int(trial.number),
        "params": dict(trial.params),
    }
    with open(cfg.HPO_BEST_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[HPO] Saved best trial → {cfg.HPO_BEST_JSON}")

def _finished_trials(study: optuna.Study):
    return [t for t in study.get_trials(deepcopy=False) if t.state.is_finished()]

def _study(storage_path: str, name: str, seed: int = 42):
    # PHASE 1 FIX: Hardened storage for NFS/HPC environments
    # Try JournalStorage first (resilient), fallback to WAL-mode SQLite
    try:
        from optuna.storages import JournalStorage, JournalFileStorage
        jpath = str(Path(storage_path).with_suffix(".journal"))
        storage = JournalStorage(JournalFileStorage(jpath))
        print(f"[HPO] Using JournalStorage: {jpath}", flush=True)
    except ImportError:
        # Fallback to SQLite with WAL mode + long timeout for reliability
        from optuna.storages import RDBStorage
        storage = RDBStorage(
            url=f"sqlite:///{storage_path}",
            engine_kwargs={
                "connect_args": {"timeout": 300, "check_same_thread": False},
                "pool_pre_ping": True,
            }
        )
        print(f"[HPO] Using RDBStorage (SQLite WAL): {storage_path}", flush=True)
    
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",  # Minimize -score (since surrogate score is higher-is-better)
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, seed=seed
        ),
        # Median pruner: prune after 6-8 epochs with patience
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=6, n_min_trials=8),
    )

def run_hpo(n_trials: int, epochs_per_trial: int, space: str = "wide", export_csv: bool = False, early_stop_patience: int = 15):
    set_seed(42)
    Path(cfg.HPO_DIR).mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_file = Path(cfg.HPO_DIR) / f"hpo_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # Tee output to both console and file
    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_f = open(log_file, 'w')
    sys.stdout = TeeLogger(sys.stdout, log_f)
    sys.stderr = TeeLogger(sys.stderr, log_f)
    
    print(f"[HPO] Logging to: {log_file}", flush=True)

    # prefer cfg.HPO_DB if you defined it, else fall back to cfg.HPO_DB_PATH
    hpo_db = getattr(cfg, "HPO_DB", str(Path(cfg.HPO_DIR) / "hpo.db"))
    study_name = getattr(cfg, "HPO_STUDY_NAME", f"L16_M64_{space}_v2_optimfix")
    study = _study(hpo_db, name=study_name, seed=42)

    print(f"[HPO] storage={hpo_db}  study={study.study_name}  space={space}", flush=True)

    def suggest(trial: optuna.Trial):
        # === TIGHTENED SEARCH SPACE ===
        if space == "wide":
            D_MODEL   = trial.suggest_categorical("D_MODEL",   [448, 512])  # Restored full model size
            NUM_HEADS = trial.suggest_categorical("NUM_HEADS", [6, 8])     # Restored full head count
            dropout   = trial.suggest_float("dropout", 0.20, 0.32)          # Tightened from [0.25, 0.45]
        else:
            D_MODEL   = trial.suggest_categorical("D_MODEL",   [448, 512])  # Restored full model size
            NUM_HEADS = trial.suggest_categorical("NUM_HEADS", [6, 8])     # Restored full head count
            dropout   = trial.suggest_float("dropout", 0.20, 0.32)

        # === Learning rate: log-uniform around baseline, ±2× range ===
        lr         = trial.suggest_float("lr", 1.0e-4, 4e-4, log=True)  # Expanded slightly from [1.5e-4, 3e-4]
        
        # Range grid: Keep moderate for HPO cost control (reserve 241 for final training)
        range_grid = trial.suggest_categorical("range_grid", [121, 161])  # HPO: [121, 161]; full training: 241
        
        # SOTA FIX: Newton refinement now runs AFTER MUSIC (not on aux head)
        # ICC FIX: Expanded ranges to support near-field Newton (sub-degree unlock)
        newton_it  = trial.suggest_int("newton_iter", 5, 15)            # MUSIC→Newton: 5-15 iters for NF support
        newton_lr  = trial.suggest_float("newton_lr", 0.3, 1.0)         # Step: 0.3-1.0 (conservative lower bound for NF)
        
        # === Loss weights (K-head removed - using MVDR peak detection) ===
        lam_cov    = trial.suggest_float("lam_cov", 0.10, 0.25)     # Covariance learning weight (primary)
        lam_ang    = trial.suggest_float("lam_ang", 0.50, 1.00)     # Angle aux weight
        lam_rng    = trial.suggest_float("lam_rng", 0.30, 0.60)     # Range aux weight
        # NOTE: lam_K removed - K estimation now done via MDL/MVDR at inference
        shrink_alpha = trial.suggest_float("shrink_alpha", 0.10, 0.20)  # Tightened from [0.10, 0.25]
        softmax_tau = trial.suggest_float("softmax_tau", 0.15, 0.25)    # Keep same
        
        # ICC CRITICAL FIX: Prefer larger batches (avoid BS=32 which degrades quickly)
        batch_size = trial.suggest_categorical("batch_size", [64, 80])  # Removed 32/48: BS=32 tanks val quickly
        
        # NF-MLE polish parameters: Limit to {0, 2} for HPO speed, reserve 3 iters for final training
        nf_mle_snr_threshold = trial.suggest_float("nf_mle_snr_threshold", 4.0, 10.0)  # dB: gate MLE on SNR
        nf_mle_iters = trial.suggest_categorical("nf_mle_iters", [0, 2])  # HPO: {0, 2}; full training: 2-3

        return dict(
            D_MODEL=D_MODEL, NUM_HEADS=NUM_HEADS, dropout=dropout,
            lr=lr, range_grid=range_grid, newton_iter=newton_it, newton_lr=newton_lr,
            lam_cov=lam_cov, lam_ang=lam_ang, lam_rng=lam_rng,
            # NOTE: lam_K removed - using MVDR peak detection
            shrink_alpha=shrink_alpha, softmax_tau=softmax_tau, batch_size=batch_size,
            nf_mle_snr_threshold=nf_mle_snr_threshold, nf_mle_iters=nf_mle_iters
        )

    def objective(trial: optuna.Trial):
        print(f"\n[HPO Trial {trial.number}] Starting trial...", flush=True)
        s = suggest(trial)
        print(f"[HPO Trial {trial.number}] Hyperparameters suggested: D_MODEL={s['D_MODEL']}, BS={s['batch_size']}, LR={s['lr']:.6f}", flush=True)

        # snapshot current mdl_cfg knobs we're about to modify
        keys = ["D_MODEL","NUM_HEADS","DROPOUT","LR_INIT","BATCH_SIZE",
                "INFERENCE_GRID_SIZE_RANGE","NEWTON_ITER","NEWTON_LR",
                "SHRINK_BASE_ALPHA","SOFTMAX_TAU","USE_EMA","USE_SWA","USE_3_PHASE_CURRICULUM"]
        snap = {k: getattr(mdl_cfg, k, None) for k in keys}

        try:
            # Disable EMA, SWA, and curriculum for HPO trials
            # CRITICAL FIX: Curriculum causes validation loss to spike when K-weight increases,
            # triggering early stopping before model can learn K properly
            mdl_cfg.USE_EMA = False
            mdl_cfg.USE_SWA = False
            mdl_cfg.USE_3_PHASE_CURRICULUM = False  # DISABLED: Curriculum sabotages HPO with early stopping
            
            # ICC CRITICAL FIX: Enable near-field Newton during HPO for proper edge-pegging fix
            # NF Newton is needed to handle the range-angle coupling correctly
            # This prevents edge pegging and improves low-SNR robustness
            cfg.NEWTON_NEARFIELD = True  # HPO: Enable NF Newton for proper 3D localization
            
            # Apply NF-MLE polish parameters (HPO-tunable for low-SNR robustness)
            cfg.NF_MLE_SNR_THRESHOLD = s["nf_mle_snr_threshold"]
            cfg.NF_MLE_ITERS = s["nf_mle_iters"]
            
            # apply suggestions - architecture
            mdl_cfg.D_MODEL   = s["D_MODEL"]
            mdl_cfg.NUM_HEADS = s["NUM_HEADS"]
            mdl_cfg.DROPOUT   = s["dropout"]
            mdl_cfg.LR_INIT   = s["lr"]
            mdl_cfg.BATCH_SIZE = s["batch_size"]
            
            # apply suggestions - inference  
            mdl_cfg.INFERENCE_GRID_SIZE_RANGE = s["range_grid"]
            mdl_cfg.NEWTON_ITER = s["newton_iter"]
            mdl_cfg.NEWTON_LR   = s["newton_lr"]
            mdl_cfg.SHRINK_BASE_ALPHA = s["shrink_alpha"]
            mdl_cfg.SOFTMAX_TAU = s["softmax_tau"]

            # Now create Trainer (after setting USE_SWA=False)
            # CRITICAL: Use SURROGATE mode for HPO (no MUSIC in validation loop)
            # This makes HPO fast, stable, and well-behaved
            cfg.VAL_PRIMARY = "surrogate"
            cfg.USE_MUSIC_METRICS_IN_VAL = False  # No MUSIC during HPO!
            print(f"[HPO Trial {trial.number}] Creating Trainer (VAL_PRIMARY={cfg.VAL_PRIMARY}, MUSIC={cfg.USE_MUSIC_METRICS_IN_VAL})...", flush=True)
            t = Trainer(from_hpo=False)
            print(f"[HPO Trial {trial.number}] Trainer created successfully", flush=True)
            
            # Populate _hpo_loss_weights so curriculum can access them
            # NOTE: lam_K removed - using MVDR peak detection
            t._hpo_loss_weights = {
                "lam_cov": s["lam_cov"],
                "lam_ang": s["lam_ang"],
                "lam_rng": s["lam_rng"],
            }
            
            # Apply loss weight suggestions (K-head removed)
            t.loss_fn.lam_cov = s["lam_cov"]         # Primary covariance weight
            t.loss_fn.lam_diag = 0.2  # 20% for diagonal (relative within NMSE)
            t.loss_fn.lam_off = 0.8   # 80% for off-diagonal (relative within NMSE)
            t.loss_fn.lam_aux = s["lam_ang"] + s["lam_rng"]  # Combined aux weight
            
            # Set reasonable defaults for missing loss parameters (not optimized by HPO)
            t.loss_fn.lam_ortho = 1e-3  # Orthogonality penalty
            t.loss_fn.lam_peak = 0.05   # Chamfer/peak angle loss
            t.loss_fn.lam_margin = 0.1  # Subspace margin regularizer
            t.loss_fn.lam_range_factor = 0.3  # Range factor in covariance
            mdl_cfg.LAM_ALIGN = 0.002  # Subspace alignment penalty

            # HPO subset: Use reasonable subset for effective HPO
            # Updated 2025-11-26: New dataset is 100K train / 10K val / 10K test
            total_train_samples = 100000  # Current L=16 dataset (100K train samples)
            total_val_samples = 10000     # Current L=16 dataset (10K val samples)
            hpo_n_train = int(0.1 * total_train_samples)  # 10K samples (10%) - effective HPO subset
            hpo_n_val = int(0.1 * total_val_samples)      # 1K samples (10%) - effective HPO subset
            
            print(f"[HPO Trial {trial.number}] Starting training with {hpo_n_train} train, {hpo_n_val} val samples...", flush=True)
            best_val = t.fit(
                epochs=epochs_per_trial,
                use_shards=True,
                n_train=hpo_n_train,    # 10K samples (10% of 100K) for effective HPO
                n_val=hpo_n_val,        # 1K samples (10% of 10K) for effective HPO
                gpu_cache=True,         # Critical for HPO speed
                grad_accumulation=1,    # REDUCED from 2 (was 256 effective batch, too smooth)
                early_stop_patience=early_stop_patience,  # Stop if no improvement for N epochs
                val_every=1,            # Validate every epoch (surrogate is fast!)
                skip_music_val=True,    # NO MUSIC during HPO - use surrogate metrics only
            )
            # CRITICAL FIX: Handle None/inf values in print
            if best_val is None or not np.isfinite(best_val):
                print(f"[HPO Trial {trial.number}] Training completed! Best objective: {best_val} (non-finite)", flush=True)
            else:
                print(f"[HPO Trial {trial.number}] Training completed! Best objective: {best_val:.6f}", flush=True)
            
            # CRITICAL FIX: Clear memory between HPO trials to prevent memory leaks
            del t
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # CRITICAL FIX: Guard against inf/nan values in HPO objective
            if not np.isfinite(best_val):
                print(f"[HPO Trial {trial.number}] Non-finite objective (val={best_val}), returning large penalty (1e6)", flush=True)
                return 1e6
            
            return float(best_val)
        finally:
            # restore mdl_cfg
            for k, v in snap.items():
                if v is None:
                    try:
                        delattr(mdl_cfg, k)
                    except Exception:
                        pass
                else:
                    setattr(mdl_cfg, k, v)

    # resume-friendly: only run the remaining trials that aren't finished yet
    done = len(_finished_trials(study))
    remaining = max(0, n_trials - done)
    print(f"[HPO] Completed trials: {done}, Remaining trials: {remaining}", flush=True)
    
    if remaining > 0:
        print(f"[HPO] Starting optimization with {remaining} trials...", flush=True)
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,
            n_jobs=1  # CRITICAL: Force sequential trials to avoid memory issues
        )
        print(f"[HPO] Optimization completed!", flush=True)

    # persist the winner
    try:
        if study.best_trial is not None:
            _save_hpo_winner(study.best_trial)
            # Note: best_value is -score (since Optuna minimizes and surrogate score is higher-is-better)
            actual_score = -study.best_value
            print(f"[HPO] Best surrogate score={actual_score:.4f} (trial={study.best_trial.number})")
    except Exception as e:
        print(f"[HPO] No completed trials yet: {e}")

    if export_csv:
        try:
            df = study.trials_dataframe()
            out = Path(cfg.HPO_DIR) / f"{study.study_name}_trials.csv"
            df.to_csv(out, index=False)
            print(f"[HPO] Exported trials CSV → {out}")
        except Exception as e:
            print(f"[HPO] CSV export skipped: {e}")




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HPO with optimized budget recommendations")
    parser.add_argument("--n_trials", type=int, default=40, 
                       help="Number of HPO trials (default: 40, recommended: 40-80)")
    parser.add_argument("--epochs_per_trial", type=int, default=12,
                       help="Epochs per trial (default: 12)")
    parser.add_argument("--space", type=str, default="wide", choices=["wide", "narrow"],
                       help="Search space (default: wide)")
    parser.add_argument("--export_csv", action="store_true",
                       help="Export results to CSV")
    
    args = parser.parse_args()
    
    print(f"🚀 HPO for SNR-Fixed Data (-5 to 20 dB):")
    print(f"  • Trials: {args.n_trials}")
    print(f"  • Epochs/trial: {args.epochs_per_trial} (with pruning)")
    print(f"  • Search space: {args.space}")
    print(f"  • Rebalanced loss weights: ✅ (reduced cov, increased ang/K)")
    print(f"  • Moderate dropout increase: ✅ (0.25-0.45 vs old 0.15-0.35)")
    print(f"  • Larger batches: ✅ (64-128 vs old 16-32)")
    print(f"  • Median pruning: ✅ Enabled")
    print()
    
    run_hpo(
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial, 
        space=args.space,
        export_csv=args.export_csv
    )
