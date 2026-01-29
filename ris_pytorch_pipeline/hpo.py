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
from .dataset import ShardNPZDataset
from .infer import hybrid_estimate_final
from .eval_angles import eval_scene_angles_ranges, hungarian_pairs_angles_ranges


def _resolve_val_dir() -> Path:
    """
    Resolve the validation shards directory.
    Prefers cfg.DATA_SHARDS_VAL; falls back to cfg.DATA_SHARDS_DIR/val.
    """
    v = Path(str(getattr(cfg, "DATA_SHARDS_VAL", "")).strip() or "")
    if str(v) and v.exists():
        return v
    # Use configured shards directory (derived from cfg.L / cfg.M_BEAMS_TARGET in configs.py).
    # Avoid hard-coded L16 fallbacks: derive from cfg if available.
    shards_dir = str(getattr(cfg, "DATA_SHARDS_DIR", "")).strip()
    if not shards_dir:
        L = int(getattr(cfg, "L", 0) or 0)
        M = int(getattr(cfg, "M_BEAMS_TARGET", 0) or 0)
        shards_dir = f"data_shards_M{M}_L{L}" if (M > 0 and L > 0) else "data_shards"
    root = Path(shards_dir)
    return root / "val"


def _ptr_to_gt_deg(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract GT (phi, theta, r) from a ShardNPZDataset item.
    Returns degrees/meters arrays of length K.
    """
    K = int(item.get("K", 0))
    KMAX = int(getattr(cfg, "K_MAX", 5))
    if "ptr" in item:
        a = item["ptr"]
        if torch.is_tensor(a):
            a = a.detach().cpu().float().reshape(-1)
            phi = torch.rad2deg(a[0:KMAX][:K]).numpy()
            tht = torch.rad2deg(a[KMAX:2 * KMAX][:K]).numpy()
            rr = a[2 * KMAX:3 * KMAX][:K].numpy()
            return phi, tht, rr
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        phi = np.rad2deg(a[0:KMAX][:K])
        tht = np.rad2deg(a[KMAX:2 * KMAX][:K])
        rr = a[2 * KMAX:3 * KMAX][:K]
        return phi, tht, rr

    # Fallback: explicit fields (radians)
    phi = np.rad2deg(np.asarray(item.get("phi", []), dtype=np.float32))
    tht = np.rad2deg(np.asarray(item.get("theta", []), dtype=np.float32))
    rr = np.asarray(item.get("r", []), dtype=np.float32)
    return phi, tht, rr


@torch.no_grad()
def _eval_mvdr_final_on_val_subset(
    model,
    *,
    n_scenes: int,
    seed: int = 0,
    blind_k: bool = True,
) -> dict:
    """
    End-to-end evaluation for HPO: run MVDR-first inference and compute production-aligned metrics.

    OBJECTIVE FUNCTION (minimize):
        objective = (rmse_xyz_all / xyz_norm) + f1_weight * (1 - F1)

    where:
        - rmse_xyz_all: RMSE of 3D Euclidean position error (meters) over ALL Hungarian pairs (ungated)
          → provides a smooth/continuous localization signal even for near-miss predictions
        - F1: 2*P*R/(P+R) with P=TP/(TP+FP), R=TP/(TP+FN)
        - TP: Hungarian-matched pairs within (phi, theta, r) tolerances
        - FP: predictions without a good match (= num_pred - TP)
        - FN: GTs without a good match (= num_gt - TP)

    Edge cases:
        - If no matched pairs: rmse_xyz_all=xyz_norm (max penalty), F1=0 → high objective
        - Perfect detection (all GTs matched, no extras): F1=1 → only localization term matters

    This directly optimizes what we care about for SOTA near-field localization:
        1. Find all sources (recall)
        2. Don't hallucinate extra sources (precision)
        3. Localize found sources accurately in 3D space (rmse_xyz)
    """
    val_dir = _resolve_val_dir()
    ds = ShardNPZDataset(val_dir)
    N = min(int(n_scenes), len(ds))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(ds), size=N, replace=False) if N < len(ds) else np.arange(len(ds))

    # Normalizers (same defaults as Trainer's MUSIC-based composite)
    phi_norm = float(getattr(cfg, "VAL_NORM_PHI_DEG", 5.0))
    theta_norm = float(getattr(cfg, "VAL_NORM_THETA_DEG", 5.0))
    r_norm = float(getattr(cfg, "VAL_NORM_R_M", 1.0))

    # Success tolerances (reuse surrogate peak tolerances by default)
    tol_phi = float(getattr(cfg, "SURROGATE_DET_TOL_PHI_DEG", 5.0))
    tol_theta = float(getattr(cfg, "SURROGATE_DET_TOL_THETA_DEG", 5.0))
    tol_r = float(getattr(cfg, "SURROGATE_DET_TOL_R_M", 1.0))

    def _sph_to_xyz(phi_deg: np.ndarray, theta_deg: np.ndarray, r_m: np.ndarray) -> np.ndarray:
        """Convert (phi, theta, r) in degrees/meters to xyz in meters (consistent for pred+gt)."""
        phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
        th = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
        r = np.asarray(r_m, dtype=np.float64)
        x = r * np.cos(th) * np.cos(phi)
        y = r * np.cos(th) * np.sin(phi)
        z = r * np.sin(th)
        return np.stack([x, y, z], axis=-1)

    rmse_phi_list: list[float] = []
    rmse_theta_list: list[float] = []
    rmse_r_list: list[float] = []
    succ = 0

    # --- Detection+localization tracking (corrected) ---
    # TP: matched pairs within tolerance
    # FP: predictions that don't match any GT within tolerance
    # FN: GTs that weren't matched within tolerance
    total_tp = 0
    total_fp = 0
    total_fn = 0
    legacy_fp = 0
    legacy_fn = 0
    total_num_gt = 0
    xyz_all_sqerr_sum = 0.0
    xyz_all_count = 0
    xyz_tp_sqerr_sum = 0.0
    xyz_tp_count = 0

    model.eval()
    for i in idx:
        it = ds[int(i)]
        gt_phi_deg, gt_th_deg, gt_r_m = _ptr_to_gt_deg(it)
        num_gt = len(gt_phi_deg)
        total_num_gt += num_gt

        # CRITICAL: H_full (true BS→RIS channel [M,N]) is required.
        H_full = it.get("H_full")
        if H_full is None:
            raise RuntimeError("HPO evaluation requires H_full in shards. Regenerate with store_h_full=True.")
        s = {
            "y": it["y"],
            "H_full": H_full,  # Use H_full instead of H_eff
            "codes": it["codes"],
            "K": int(it["K"]),
            "snr_db": float(it["snr"]),
        }
        force_K = None if bool(blind_k) else int(it["K"])

        try:
            ph, tht, rr = hybrid_estimate_final(
                model,
                s,
                force_K=force_K,
                k_policy="mdl",
                do_newton=bool(getattr(cfg, "NEWTON_NEARFIELD", True)),
                use_hpo_knobs=False,  # inference knobs no longer HPO-controlled
            )
        except Exception:
            ph, tht, rr = ([], [], [])

        ph = np.rad2deg(np.asarray(ph, dtype=np.float32))
        tht = np.rad2deg(np.asarray(tht, dtype=np.float32))
        rr = np.asarray(rr, dtype=np.float32)
        num_pred = len(ph)

        err = eval_scene_angles_ranges(
            ph,
            tht,
            rr,
            gt_phi_deg,
            gt_th_deg,
            gt_r_m,
            success_tol_phi=tol_phi,
            success_tol_theta=tol_theta,
            success_tol_r=tol_r,
        )

        # Penalize empty/invalid preds by using norm-scale errors (for legacy metrics)
        rmse_phi_list.append(float(err["rmse_phi"]) if err["rmse_phi"] is not None else phi_norm)
        rmse_theta_list.append(float(err["rmse_theta"]) if err["rmse_theta"] is not None else theta_norm)
        rmse_r_list.append(float(err["rmse_r"]) if err["rmse_r"] is not None else r_norm)

        succ += int(bool(err.get("success_flag", False)))

        # Legacy FP/FN based on count difference (kept for backward-comparable metric only).
        legacy_fp += max(0, num_pred - num_gt)
        legacy_fn += max(0, num_gt - num_pred)

        # --- CORRECTED TP/FP/FN: tolerance-gated Hungarian matching ---
        # TP = matched pairs within tolerance
        # FP = num_pred - TP (predictions without good match)
        # FN = num_gt - TP (GTs without good match)
        pairs = hungarian_pairs_angles_ranges(ph, tht, rr, gt_phi_deg, gt_th_deg, gt_r_m)
        tp_scene = 0
        for pi, gi in pairs:
            dphi = float(abs(ph[pi] - gt_phi_deg[gi]))
            dth = float(abs(tht[pi] - gt_th_deg[gi]))
            dr = float(abs(rr[pi] - gt_r_m[gi]))
            ok = (dphi <= tol_phi) and (dth <= tol_theta) and (dr <= tol_r)

            # Always accumulate ungated xyz error (continuity for objective).
            p_xyz = _sph_to_xyz(ph[pi], tht[pi], rr[pi])
            g_xyz = _sph_to_xyz(gt_phi_deg[gi], gt_th_deg[gi], gt_r_m[gi])
            de = float(np.linalg.norm(p_xyz - g_xyz))
            xyz_all_sqerr_sum += de * de
            xyz_all_count += 1

            if ok:
                # Also track TP-only xyz error for diagnostics.
                xyz_tp_sqerr_sum += de * de
                xyz_tp_count += 1
                tp_scene += 1

        # Correct FP/FN based on TP, not count difference
        total_tp += tp_scene
        total_fp += max(0, num_pred - tp_scene)  # Extra predictions
        total_fn += max(0, num_gt - tp_scene)    # Missed GTs

    rmse_phi_mean = float(np.mean(rmse_phi_list)) if rmse_phi_list else float("inf")
    rmse_theta_mean = float(np.mean(rmse_theta_list)) if rmse_theta_list else float("inf")
    rmse_r_mean = float(np.mean(rmse_r_list)) if rmse_r_list else float("inf")
    success_rate = float(succ) / max(1.0, float(N))
    fp_per_scene = float(total_fp) / max(1.0, float(N))
    fn_per_scene = float(total_fn) / max(1.0, float(N))
    fp_per_scene_legacy = float(legacy_fp) / max(1.0, float(N))
    fn_per_scene_legacy = float(legacy_fn) / max(1.0, float(N))

    # Legacy composite (kept for reporting/comparison)
    objective_legacy = (rmse_phi_mean / phi_norm) + (rmse_theta_mean / theta_norm) + (rmse_r_mean / r_norm) - success_rate
    objective_legacy += 0.25 * fp_per_scene_legacy + 0.25 * fn_per_scene_legacy

    # --- Production-aligned objective (default): detection quality + meter-domain localization ---
    # Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2PR/(P+R)
    precision = float(total_tp) / max(1.0, float(total_tp + total_fp))
    recall = float(total_tp) / max(1.0, float(total_tp + total_fn))
    f1 = (2.0 * precision * recall) / max(1e-12, (precision + recall))

    # xyz RMSE over ALL matched pairs (ungated): smooth, continuous objective signal.
    xyz_norm = float(getattr(cfg, "HPO_E2E_XYZ_NORM_M", 0.5))
    if xyz_all_count > 0:
        rmse_xyz_mean = float(np.sqrt(xyz_all_sqerr_sum / float(xyz_all_count)))
    else:
        # No matched pairs at all → max penalty (= norm, so normalized term = 1.0)
        rmse_xyz_mean = xyz_norm

    # TP-only xyz RMSE (diagnostic, reflects localization quality when detection is correct)
    if xyz_tp_count > 0:
        rmse_xyz_tp = float(np.sqrt(xyz_tp_sqerr_sum / float(xyz_tp_count)))
    else:
        rmse_xyz_tp = float("inf")

    f1_w = float(getattr(cfg, "HPO_E2E_F1_WEIGHT", 2.0))
    objective_xyz_f1 = (rmse_xyz_mean / max(1e-12, xyz_norm)) + f1_w * (1.0 - f1)

    mode = str(getattr(cfg, "HPO_E2E_OBJECTIVE", "xyz_f1")).lower().strip()
    objective = objective_xyz_f1 if mode != "legacy" else float(objective_legacy)

    return dict(
        rmse_phi_mean=rmse_phi_mean,
        rmse_theta_mean=rmse_theta_mean,
        rmse_r_mean=rmse_r_mean,
        success_rate=success_rate,
        fp_per_scene=fp_per_scene,
        fn_per_scene=fn_per_scene,
        fp_per_scene_legacy=fp_per_scene_legacy,
        fn_per_scene_legacy=fn_per_scene_legacy,
        # Detection metrics (corrected)
        rmse_xyz_mean=rmse_xyz_mean,
        rmse_xyz_tp=rmse_xyz_tp,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=int(total_tp),
        fp=int(total_fp),
        fn=int(total_fn),
        total_gt=int(total_num_gt),
        objective_legacy=float(objective_legacy),
        objective=float(objective),
        n_scenes=int(N),
    )

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


def _completed_trials_sorted(study: optuna.Study):
    """Return COMPLETE trials sorted by objective value (ascending)."""
    ts = []
    for t in study.get_trials(deepcopy=False):
        if t.state == optuna.trial.TrialState.COMPLETE and (t.value is not None) and np.isfinite(float(t.value)):
            ts.append(t)
    ts.sort(key=lambda x: float(x.value))
    return ts


def run_hpo(
    n_trials: int,
    epochs_per_trial: int,
    space: str = "wide",
    export_csv: bool = False,
    early_stop_patience: int = 15,
    *,
    objective_mode: str = "surrogate",
    e2e_val_scenes: int | None = None,
    e2e_seed: int = 0,
    e2e_blind_k: bool = True,
    enqueue_params: list[dict] | None = None,
    study_name_override: str | None = None,
):
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

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    log_f = open(log_file, "w")
    sys.stdout = TeeLogger(orig_stdout, log_f)
    sys.stderr = TeeLogger(orig_stderr, log_f)

    try:
        print(f"[HPO] Logging to: {log_file}", flush=True)

        # prefer cfg.HPO_DB if you defined it, else fall back to cfg.HPO_DB_PATH
        hpo_db = getattr(cfg, "HPO_DB", str(Path(cfg.HPO_DIR) / "hpo.db"))

        # Use distinct study names so results don't mix between objectives.
        mode = str(objective_mode).lower().strip()
        suffix = "surrogate" if mode == "surrogate" else "mvdr_final"
        # Derive study naming from system geometry to avoid mixing studies across L/M.
        L_tag = f"L{int(getattr(cfg, 'L', 0))}"
        M_tag = f"M{int(getattr(cfg, 'M_BEAMS_TARGET', 0))}"
        study_name = study_name_override or getattr(cfg, "HPO_STUDY_NAME", f"{L_tag}_{M_tag}_{space}_v2_{suffix}")
        study = _study(hpo_db, name=study_name, seed=42)

        # Optional: enqueue a fixed list of parameter dicts (used by two-stage rerank).
        if enqueue_params:
            for p in enqueue_params:
                if isinstance(p, dict) and p:
                    try:
                        study.enqueue_trial(p)
                    except Exception:
                        pass

        print(f"[HPO] storage={hpo_db}  study={study.study_name}  space={space}", flush=True)

        def suggest(trial: optuna.Trial):
            # === TIGHTENED SEARCH SPACE ===
            if space == "wide":
                D_MODEL = trial.suggest_categorical("D_MODEL", [448, 512])  # Restored full model size
                NUM_HEADS = trial.suggest_categorical("NUM_HEADS", [6, 8])  # Restored full head count
                dropout = trial.suggest_float("dropout", 0.20, 0.32)  # Tightened from [0.25, 0.45]
            else:
                D_MODEL = trial.suggest_categorical("D_MODEL", [448, 512])  # Restored full model size
                NUM_HEADS = trial.suggest_categorical("NUM_HEADS", [6, 8])  # Restored full head count
                dropout = trial.suggest_float("dropout", 0.20, 0.32)

            # === Learning rate: log-uniform around baseline, ±2× range ===
            lr = trial.suggest_float("lr", 1.0e-4, 4e-4, log=True)  # Expanded slightly from [1.5e-4, 3e-4]

            # === Loss weights ===
            # IMPORTANT: MVDR-first inference depends primarily on learning a good covariance.
            # The previous ranges under-weighted covariance relative to aux heads and led to
            # uniformly-bad MVDR-final objectives (F1≈0, rmse_xyz≈4m).
            #
            # New policy (production-aligned):
            # - lam_cov is dominant
            # - aux heads are mild regularizers (still useful for stability), but not the driver
            lam_cov = trial.suggest_float("lam_cov", 0.5, 2.0)   # covariance NMSE weight (primary)
            lam_ang = trial.suggest_float("lam_ang", 0.0, 0.30)  # aux angle weight (regularizer)
            lam_rng = trial.suggest_float("lam_rng", 0.0, 0.30)  # aux range weight (regularizer)
            # Shrinkage "base alpha" used by torch/np shrink utilities:
            #   alpha = base_alpha * 10^(-SNR/20), clamped to <= 5e-2.
            # For L=64, we generally want *less* shrinkage than L=16. Search in a log range.
            shrink_base_alpha = trial.suggest_float("shrink_base_alpha", 5e-4, 2e-2, log=True)
            softmax_tau = trial.suggest_float("softmax_tau", 0.15, 0.25)

            # Prefer larger batches (avoid BS=32 which degrades quickly)
            batch_size = trial.suggest_categorical("batch_size", [64, 80])

            # NOTE: hybrid_cov_beta removed from HPO — R_samp is broken (F1=0 on validation).
            # Once R_samp is fixed, re-enable: hybrid_beta = trial.suggest_float("hybrid_cov_beta", 0.15, 0.55)

            # NF-MLE polish parameters
            nf_mle_snr_threshold = trial.suggest_float("nf_mle_snr_threshold", 4.0, 10.0)
            nf_mle_iters = trial.suggest_categorical("nf_mle_iters", [0, 2])

            return dict(
                D_MODEL=D_MODEL,
                NUM_HEADS=NUM_HEADS,
                dropout=dropout,
                lr=lr,
                lam_cov=lam_cov,
                lam_ang=lam_ang,
                lam_rng=lam_rng,
                shrink_base_alpha=shrink_base_alpha,
                softmax_tau=softmax_tau,
                batch_size=batch_size,
                nf_mle_snr_threshold=nf_mle_snr_threshold,
                nf_mle_iters=nf_mle_iters,
            )

        def objective(trial: optuna.Trial):
            print(f"\n[HPO Trial {trial.number}] Starting trial...", flush=True)
            s = suggest(trial)
            print(
                f"[HPO Trial {trial.number}] Hyperparameters suggested: D_MODEL={s['D_MODEL']}, BS={s['batch_size']}, LR={s['lr']:.6f}",
                flush=True,
            )

            mode_local = str(objective_mode).lower().strip()
            if mode_local != "surrogate":
                print(
                    f"[HPO Trial {trial.number}] NOTE: objective_mode={mode_local}. "
                    f"Training/early-stop uses fast surrogate validation; MVDR-final E2E scoring runs ONCE after training.",
                    flush=True,
                )

            # snapshot current mdl_cfg knobs we're about to modify
            keys = [
                "D_MODEL",
                "NUM_HEADS",
                "DROPOUT",
                "LR_INIT",
                "BATCH_SIZE",
                "SHRINK_BASE_ALPHA",
                "SOFTMAX_TAU",
                "USE_EMA",
                "USE_SWA",
                "USE_3_PHASE_CURRICULUM",
            ]
            snap = {k: getattr(mdl_cfg, k, None) for k in keys}

            try:
                # Mark HPO mode for Trainer (fail-fast on NaN grads, disable unstable loss terms by default)
                cfg.HPO_MODE = True

                # Disable EMA, SWA, and curriculum for HPO trials
                mdl_cfg.USE_EMA = False
                mdl_cfg.USE_SWA = False
                mdl_cfg.USE_3_PHASE_CURRICULUM = False

                # Enable near-field Newton during HPO for proper 3D localization
                cfg.NEWTON_NEARFIELD = True

                # Apply NF-MLE polish parameters (HPO-tunable for low-SNR robustness)
                cfg.NF_MLE_SNR_THRESHOLD = s["nf_mle_snr_threshold"]
                cfg.NF_MLE_ITERS = s["nf_mle_iters"]

                # Hybrid covariance blend: DISABLED because R_samp is broken (F1=0).
                # Once R_samp is fixed, re-enable HPO tuning of this parameter.
                cfg.HYBRID_COV_BETA = 0.0

                # apply suggestions - architecture
                mdl_cfg.D_MODEL = s["D_MODEL"]
                mdl_cfg.NUM_HEADS = s["NUM_HEADS"]
                mdl_cfg.DROPOUT = s["dropout"]
                mdl_cfg.LR_INIT = s["lr"]
                mdl_cfg.BATCH_SIZE = s["batch_size"]

                # apply suggestions - conditioning / calibration knobs
                mdl_cfg.SHRINK_BASE_ALPHA = s["shrink_base_alpha"]
                mdl_cfg.SOFTMAX_TAU = s["softmax_tau"]

                # Now create Trainer (after setting USE_SWA=False)
                # IMPORTANT: Even for objective_mode="mvdr_final", we keep Trainer validation in fast "surrogate"
                # mode (MUSIC-free) for speed. MVDR-final is evaluated once at the end of the trial.
                cfg.VAL_PRIMARY = "surrogate"
                cfg.USE_MUSIC_METRICS_IN_VAL = False  # No MUSIC during HPO!
                print(
                    f"[HPO Trial {trial.number}] Creating Trainer (VAL_PRIMARY={cfg.VAL_PRIMARY}, MUSIC={cfg.USE_MUSIC_METRICS_IN_VAL}, objective_mode={mode_local})...",
                    flush=True,
                )
                t = Trainer(from_hpo=False)
                print(f"[HPO Trial {trial.number}] Trainer created successfully", flush=True)

                # Populate _hpo_loss_weights so curriculum can access them
                t._hpo_loss_weights = {
                    "lam_cov": s["lam_cov"],
                    "lam_ang": s["lam_ang"],
                    "lam_rng": s["lam_rng"],
                }

                # Apply loss weight suggestions
                t.loss_fn.lam_cov = s["lam_cov"]
                t.loss_fn.lam_diag = 0.2
                t.loss_fn.lam_off = 0.8
                t.loss_fn.lam_aux = s["lam_ang"] + s["lam_rng"]

                # Stable defaults for non-optimized loss params
                t.loss_fn.lam_ortho = 1e-3
                t.loss_fn.lam_peak = 0.05
                if bool(getattr(cfg, "HPO_DISABLE_UNSTABLE_LOSS_TERMS", True)):
                    t.loss_fn.lam_gap = 0.0
                    t.loss_fn.lam_margin = 0.0
                    t.loss_fn.lam_subspace_align = 0.0
                    t.loss_fn.lam_peak_contrast = 0.0
                    mdl_cfg.LAM_ALIGN = 0.0
                else:
                    t.loss_fn.lam_gap = 0.0
                    t.loss_fn.lam_margin = 0.0
                t.loss_fn.lam_range_factor = 0.3
                if not bool(getattr(cfg, "HPO_DISABLE_UNSTABLE_LOSS_TERMS", True)):
                    mdl_cfg.LAM_ALIGN = 0.002

                # HPO subset: 10K train / 1K val (10% each)
                total_train_samples = 100000
                total_val_samples = 10000
                hpo_n_train = int(0.1 * total_train_samples)
                hpo_n_val = int(0.1 * total_val_samples)

                # Stage-1 surrogate: enable a cheap MVDR-peak proxy to make the surrogate correlate with MVDR-final.
                # This does NOT run the full E2E objective; it just computes peak-level P/R/F1 on a small cap.
                # (Stage-2 still uses the true MVDR-final objective at the end of each trial.)
                if str(getattr(cfg, "VAL_PRIMARY", "surrogate")).lower().strip() == "surrogate":
                    cfg.SURROGATE_PEAK_METRICS = True
                    cfg.SURROGATE_PEAK_MAX_SCENES = int(getattr(cfg, "HPO_SURROGATE_PEAK_MAX_SCENES", 64))
                    cfg.SURROGATE_PEAK_GRID_PHI = int(getattr(cfg, "HPO_SURROGATE_PEAK_GRID_PHI", 121))
                    cfg.SURROGATE_PEAK_GRID_THETA = int(getattr(cfg, "HPO_SURROGATE_PEAK_GRID_THETA", 61))
                    # Default weights: prioritize peak F1, lightly penalize FP/scene, and keep loss as a stabilizer.
                    cfg.SURROGATE_METRIC_WEIGHTS = getattr(cfg, "SURROGATE_METRIC_WEIGHTS", None) or {
                        "w_loss": 0.5,
                        "w_aux_ang": 0.002,
                        "w_aux_r": 0.002,
                        "w_peak_f1": 6.0,
                        "w_peak_fp": 0.25,
                        "w_peak_pssr": 0.0,
                    }

                print(
                    f"[HPO Trial {trial.number}] Starting training with {hpo_n_train} train, {hpo_n_val} val samples...",
                    flush=True,
                )
                try:
                    best_val = t.fit(
                        epochs=epochs_per_trial,
                        use_shards=True,
                        n_train=hpo_n_train,
                        n_val=hpo_n_val,
                        gpu_cache=True,
                        grad_accumulation=1,
                        early_stop_patience=early_stop_patience,
                        val_every=1,
                        skip_music_val=True,
                    )
                except RuntimeError as e:
                    msg = str(e)
                    if "Non-finite gradients detected" in msg:
                        raise optuna.TrialPruned(msg)
                    raise

                # Optional MVDR-first end-to-end scoring (production-aligned objective)
                final_obj = float(best_val) if best_val is not None else float("inf")
                if mode_local != "surrogate":
                    try:
                        best_path = Path(cfg.CKPT_DIR) / "best.pt"
                        if best_path.exists():
                            sd = torch.load(str(best_path), map_location="cpu", weights_only=False)
                            t.model.load_state_dict(sd, strict=False)

                        n_eval = (
                            int(e2e_val_scenes)
                            if (e2e_val_scenes is not None)
                            else int(getattr(cfg, "HPO_E2E_VAL_SCENES", 500))
                        )
                        e2e = _eval_mvdr_final_on_val_subset(
                            t.model,
                            n_scenes=n_eval,
                            seed=int(e2e_seed),
                            blind_k=bool(e2e_blind_k),
                        )
                        final_obj = float(e2e["objective"])
                        for k, v in e2e.items():
                            try:
                                trial.set_user_attr(f"e2e_{k}", v)
                            except Exception:
                                pass
                        print(
                            f"[HPO Trial {trial.number}] E2E(MVDR-final) obj={final_obj:.4f} "
                            f"rmse_xyz={e2e.get('rmse_xyz_mean', float('nan')):.3f}m "
                            f"F1={e2e.get('f1', 0):.3f} (P={e2e.get('precision', 0):.3f} R={e2e.get('recall', 0):.3f}) "
                            f"TP={e2e.get('tp', 0)}/{e2e.get('total_gt', 0)} "
                            f"FP={e2e.get('fp', 0)} FN={e2e.get('fn', 0)}",
                            flush=True,
                        )
                    except Exception as e:
                        print(
                            f"[HPO Trial {trial.number}] E2E eval failed: {e} -> returning large penalty (1e6)",
                            flush=True,
                        )
                        final_obj = 1e6

                if final_obj is None or not np.isfinite(final_obj):
                    print(
                        f"[HPO Trial {trial.number}] Training completed! Best objective: {final_obj} (non-finite)",
                        flush=True,
                    )
                else:
                    print(f"[HPO Trial {trial.number}] Training completed! Best objective: {final_obj:.6f}", flush=True)

                # Clear memory between HPO trials
                del t
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not np.isfinite(final_obj):
                    print(
                        f"[HPO Trial {trial.number}] Non-finite objective (val={final_obj}), returning large penalty (1e6)",
                        flush=True,
                    )
                    return 1e6

                return float(final_obj)
            finally:
                cfg.HPO_MODE = False
                for k, v in snap.items():
                    if v is None:
                        try:
                            delattr(mdl_cfg, k)
                        except Exception:
                            pass
                    else:
                        setattr(mdl_cfg, k, v)

        # resume-friendly: only run remaining trials
        done = len(_finished_trials(study))
        remaining = max(0, int(n_trials) - done)
        print(f"[HPO] Completed trials: {done}, Remaining trials: {remaining}", flush=True)

        if remaining > 0:
            print(f"[HPO] Starting optimization with {remaining} trials...", flush=True)
            study.optimize(
                objective,
                n_trials=remaining,
                catch=(RuntimeError,),
                gc_after_trial=True,
                show_progress_bar=False,
                n_jobs=1,
            )
            print("[HPO] Optimization completed!", flush=True)

        # persist the winner
        try:
            if study.best_trial is not None:
                _save_hpo_winner(study.best_trial)
                if mode == "surrogate":
                    actual_score = -study.best_value
                    print(f"[HPO] Best surrogate score={actual_score:.4f} (trial={study.best_trial.number})")
                else:
                    print(f"[HPO] Best MVDR-final objective={study.best_value:.6f} (trial={study.best_trial.number})")
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
    finally:
        try:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
        except Exception:
            pass
        try:
            log_f.close()
        except Exception:
            pass


def run_hpo_two_stage(
    *,
    stage1_trials: int,
    stage1_epochs: int,
    stage2_topk: int,
    stage2_epochs: int,
    space: str = "wide",
    early_stop_patience: int = 15,
    e2e_val_scenes: int = 1000,
    e2e_seed: int = 0,
    e2e_blind_k: bool = True,
):
    """
    Two-stage HPO:
      - Stage 1: many trials, fast surrogate objective
      - Stage 2: rerank top-K parameter sets using MVDR-first end-to-end objective
        (re-trains each candidate, then evaluates via hybrid_estimate_final on a fixed val subset).
    """
    hpo_db = getattr(cfg, "HPO_DB", str(Path(cfg.HPO_DIR) / "hpo.db"))
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Stage 1
    L_tag = f"L{int(getattr(cfg, 'L', 0))}"
    M_tag = f"M{int(getattr(cfg, 'M_BEAMS_TARGET', 0))}"
    s1_name = f"{L_tag}_{M_tag}_{space}_2stage_{stamp}_stage1_surrogate"
    run_hpo(
        n_trials=int(stage1_trials),
        epochs_per_trial=int(stage1_epochs),
        space=space,
        export_csv=True,
        early_stop_patience=early_stop_patience,
        objective_mode="surrogate",
        study_name_override=s1_name,
    )

    # Reload study1 and pick top-K
    study1 = _study(hpo_db, name=s1_name, seed=42)
    top = _completed_trials_sorted(study1)[: max(1, int(stage2_topk))]
    enqueue = [dict(t.params) for t in top if isinstance(t.params, dict)]

    print(f"[HPO2] Stage1 best value={float(study1.best_value):.6f} (trial={study1.best_trial.number})", flush=True)
    print(f"[HPO2] Reranking top-K={len(enqueue)} via MVDR-final objective...", flush=True)

    # Stage 2
    s2_name = f"{L_tag}_{M_tag}_{space}_2stage_{stamp}_stage2_mvdr_final"
    run_hpo(
        n_trials=len(enqueue),
        epochs_per_trial=int(stage2_epochs),
        space=space,
        export_csv=True,
        early_stop_patience=early_stop_patience,
        objective_mode="mvdr_final",
        e2e_val_scenes=int(e2e_val_scenes),
        e2e_seed=int(e2e_seed),
        e2e_blind_k=bool(e2e_blind_k),
        enqueue_params=enqueue,
        study_name_override=s2_name,
    )


def run_hpo_stage2_rerank_from_stage1(
    *,
    stage1_study_name: str,
    stage2_topk: int,
    stage2_epochs: int,
    stage2_study_name: str | None = None,
    space: str = "wide",
    early_stop_patience: int = 15,
    e2e_val_scenes: int = 1000,
    e2e_seed: int = 0,
    e2e_blind_k: bool = True,
):
    """
    Stage-2-only rerank utility.

    Use this when Stage-1 was already completed (possibly on another machine) and you want
    to rerank Stage-1 top-K trials via the MVDR-final end-to-end objective.

    This reads Stage-1 trials from the same Optuna storage (cfg.HPO_DB -> *.journal) and
    enqueues the top-K parameter dicts into a new Stage-2 study.
    """
    if not stage1_study_name or (not str(stage1_study_name).strip()):
        raise ValueError("stage1_study_name is required")

    hpo_db = getattr(cfg, "HPO_DB", str(Path(cfg.HPO_DIR) / "hpo.db"))
    stage1_study_name = str(stage1_study_name).strip()

    # Load stage-1 study and pick top-K completed trials
    study1 = _study(hpo_db, name=stage1_study_name, seed=42)
    top = _completed_trials_sorted(study1)[: max(1, int(stage2_topk))]
    enqueue = [dict(t.params) for t in top if isinstance(t.params, dict)]

    if not enqueue:
        raise RuntimeError(
            f"No completed trials found in stage1 study '{stage1_study_name}'. "
            f"Check the storage path ({hpo_db}) and study name."
        )

    # Default stage-2 name: derived from stage-1 name (stable across machines)
    s2_name = str(stage2_study_name).strip() if stage2_study_name else f"{stage1_study_name}_stage2_mvdr_final"

    print(f"[HPO2] Stage2-only rerank from stage1 study='{stage1_study_name}'", flush=True)
    print(f"[HPO2] Stage1 best value={float(study1.best_value):.6f} (trial={study1.best_trial.number})", flush=True)
    print(f"[HPO2] Reranking top-K={len(enqueue)} into stage2 study='{s2_name}'", flush=True)

    run_hpo(
        n_trials=len(enqueue),
        epochs_per_trial=int(stage2_epochs),
        space=space,
        export_csv=True,
        early_stop_patience=early_stop_patience,
        objective_mode="mvdr_final",
        e2e_val_scenes=int(e2e_val_scenes),
        e2e_seed=int(e2e_seed),
        e2e_blind_k=bool(e2e_blind_k),
        enqueue_params=enqueue,
        study_name_override=s2_name,
    )




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
