import argparse
from pathlib import Path
from .configs import cfg, mdl_cfg, set_seed
from .dataset import prepare_shards, prepare_split_shards, set_sampling_overrides_from_cfg
from .train import Trainer

def main():
    parser = argparse.ArgumentParser("RIS PyTorch Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # --- flat pregen (single folder) ---
    g1 = sub.add_parser("pregen", help="Pre-generate NPZ shards (single folder)")
    g1.add_argument("--n", type=int, default=200000)
    g1.add_argument("--shard", type=int, default=25000)
    g1.add_argument("--eta", type=float, default=0.05)
    g1.add_argument("--L", type=int, default=None)
    g1.add_argument("--seed", type=int, default=42)
    g1.add_argument("--out_dir", type=str, default=str(getattr(cfg, "DATA_SHARDS_DIR", "results_final/data/shards")))

    # robust-track knobs for pregen
    for g in (g1,):
        g.add_argument("--phi-fov-deg", type=float, default=60.0)
        g.add_argument("--theta-fov-deg", type=float, default=30.0)
        g.add_argument("--r-min", type=float, default=0.5)
        g.add_argument("--r-max", type=float, default=10.0)
        g.add_argument("--snr-min", type=float, default=-5.0)
        g.add_argument("--snr-max", type=float, default=20.0)
        g.add_argument("--grid-offset", action="store_true")
        g.add_argument("--grid-offset-frac", type=float, default=0.5)
        g.add_argument("--dr", action="store_true")
        g.add_argument("--dr-phase-sigma", type=float, default=0.04)
        g.add_argument("--dr-amp-jitter", type=float, default=0.05)
        g.add_argument("--dr-dropout-p", type=float, default=0.03)
        g.add_argument("--dr-wav-jitter", type=float, default=0.002)
        g.add_argument("--dr-dspacing-jitter", type=float, default=0.005)

    # --- split pregen (train/val/test subfolders) ---
    g1s = sub.add_parser("pregen-split", help="Pre-generate train/val/test splits")
    g1s.add_argument("--n-train", type=int, default=160000)
    g1s.add_argument("--n-val",   type=int, default=40000)
    g1s.add_argument("--n-test",  type=int, default=8000)
    g1s.add_argument("--shard", type=int, default=25000)
    g1s.add_argument("--eta", type=float, default=0.05)
    g1s.add_argument("--L", type=int, default=None)
    g1s.add_argument("--seed", type=int, default=42)
    g1s.add_argument("--out_dir", type=str, default=str(getattr(cfg, "DATA_SHARDS_DIR", "results_final/data/shards")))
    # same robust args
    for g in (g1s,):
        g.add_argument("--phi-fov-deg", type=float, default=60.0)
        g.add_argument("--theta-fov-deg", type=float, default=30.0)
        g.add_argument("--r-min", type=float, default=0.5)
        g.add_argument("--r-max", type=float, default=10.0)
        g.add_argument("--snr-min", type=float, default=-5.0)
        g.add_argument("--snr-max", type=float, default=20.0)
        g.add_argument("--grid-offset", action="store_true")
        g.add_argument("--grid-offset-frac", type=float, default=0.5)
        g.add_argument("--dr", action="store_true")
        g.add_argument("--dr-phase-sigma", type=float, default=0.04)
        g.add_argument("--dr-amp-jitter", type=float, default=0.05)
        g.add_argument("--dr-dropout-p", type=float, default=0.03)
        g.add_argument("--dr-wav-jitter", type=float, default=0.002)
        g.add_argument("--dr-dspacing-jitter", type=float, default=0.005)

    # --- train ---
    g2 = sub.add_parser("train", help="Train model")
    g2.add_argument("--epochs", type=int, default=mdl_cfg.EPOCHS)
    g2.add_argument("--n_train", type=int, default=160000)
    g2.add_argument("--n_val", type=int, default=40000)
    g2.add_argument("--use_shards", action="store_true")
    g2.add_argument("--from_hpo", type=str, default=None,
                    help="Path to results_final/hpo/best.json")

    # --- HPO ---
    g = sub.add_parser("hpo", help="Hyperparameter optimization")
    g.add_argument("--trials", type=int, default=60)
    g.add_argument("--hpo-epochs", type=int, default=40)  # ICC FIX: 40 epochs for proper exploration
    g.add_argument("--space", choices=["small","wide","xl"], default="wide")
    g.add_argument("--early-stop-patience", type=int, default=15, help="Early stopping patience (epochs)")  # ICC FIX: patience=15

    # --- quick bench (legacy scatter sanity, writes gt_phi0 vs pred_phi0) ---
    g_bench = sub.add_parser("bench", help="Quick bench: writes pairs [gt_phi0, pred_phi0] CSV")
    g_bench.add_argument("--tag", default="quick", help="CSV filename stem under results_final/benches/")
    g_bench.add_argument("--n", type=int, default=50, help="Max samples")

    # --- full benchmark suite (ICC battery) ---
    g_suite = sub.add_parser("suite", help="Run ICC benchmark battery (B1..B9)")
    g_suite.add_argument("--oracle_too", action="store_true", help="Also run an all-oracle pass")
    g_suite.add_argument("--include_decoupled", action="store_true", help="Include Decoupled-MOD-MUSIC baseline")
    g_suite.add_argument("--limit", type=int, default=None, help="Limit per sweep (debug)")

    # --- plots/tables/export/doctor (optional) ---
    sub.add_parser("plots", help="Render standard benchmark plots")
    sub.add_parser("latex", help="Generate LaTeX tables from CSVs")
    sub.add_parser("export", help="Package tables+figures into paper/ and emit main_results.tex")
    sub.add_parser("doctor", help="Run environment & dataset checks")

    # targeted bench selector (optional; lets you run just one slice)
    g_suite.add_argument(
        "--bench", default=None,
        help="Pick one: B1, B2, B3, B4, B5, B6, B7, B4_k2_snr_sweep, B8_rmse_vs_K_at_snr, B9_heatmap_K_by_SNR"
    )
    g_suite.add_argument("--snr", type=float, default=15.0, help="Target SNR for B8 (± tol)")
    g_suite.add_argument("--tol", type=float, default=0.75, help="Tolerance around target SNR for B8")


    args = parser.parse_args()
    Path(cfg.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    # Keep all pipeline artifacts under cfg.RESULTS_DIR for consistency.
    Path(cfg.RESULTS_DIR, "models").mkdir(parents=True, exist_ok=True)
    Path(cfg.RESULTS_DIR, "benches").mkdir(parents=True, exist_ok=True)
    Path(cfg.RESULTS_DIR, "figs").mkdir(parents=True, exist_ok=True)

    # ---- command router ----
    if args.cmd in ("pregen","pregen-split"):
        # Push CLI robust options into mdl_cfg, then enable overrides for *pregen only*
        mdl_cfg.TRAIN_PHI_FOV_DEG   = getattr(args, "phi_fov_deg")
        mdl_cfg.TRAIN_THETA_FOV_DEG = getattr(args, "theta_fov_deg")
        mdl_cfg.TRAIN_R_MIN_MAX     = (getattr(args, "r_min"), getattr(args, "r_max"))
        
        # Drive SNR directly during pregen (fixes SNR distribution issue)
        mdl_cfg.SNR_DB_RANGE        = (getattr(args, "snr_min"), getattr(args, "snr_max"))
        mdl_cfg.SNR_TARGETED        = True
        # (Optionally still keep these for the legacy non-targeted path)
        mdl_cfg.P_DB_RANGE          = cfg.P_DB_RANGE
        mdl_cfg.NOISE_STD_RANGE     = cfg.NOISE_STD_RANGE
        
        mdl_cfg.GRID_OFFSET_ENABLED = bool(getattr(args, "grid_offset"))
        mdl_cfg.GRID_OFFSET_FRAC    = getattr(args, "grid_offset_frac")
        mdl_cfg.DR_ENABLED          = bool(getattr(args, "dr"))
        mdl_cfg.DR_PHASE_SIGMA      = getattr(args, "dr_phase_sigma")
        mdl_cfg.DR_AMP_JITTER       = getattr(args, "dr_amp_jitter")
        mdl_cfg.DR_DROPOUT_P        = getattr(args, "dr_dropout_p")
        mdl_cfg.DR_WAVELEN_JITTER   = getattr(args, "dr_wav_jitter")
        mdl_cfg.DR_DSPACING_JITTER  = getattr(args, "dr_dspacing_jitter")
        set_sampling_overrides_from_cfg(mdl_cfg)

        if args.cmd == "pregen":
            prepare_shards(Path(args.out_dir), n_samples=args.n,
                           shard_size=args.shard, seed=args.seed,
                           eta_perturb=args.eta, override_L=args.L)
        else:
            prepare_split_shards(Path(args.out_dir),
                                 n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
                                 shard_size=args.shard, seed=args.seed,
                                 eta_perturb=args.eta, override_L=args.L)

    elif args.cmd == "train":
        t = Trainer(from_hpo=args.from_hpo)
        t.fit(epochs=args.epochs, n_train=args.n_train, n_val=args.n_val, use_shards=args.use_shards)

    elif args.cmd == "hpo":
        from .hpo import run_hpo
        run_hpo(n_trials=args.trials, epochs_per_trial=args.hpo_epochs, space=args.space, export_csv=True, early_stop_patience=args.early_stop_patience)

    elif args.cmd == "bench":
        # Quick sanity bench (scatter CSV)
        from .infer import load_model
        from .benchmark import run_bench_csv
        m = load_model()
        out_csv = Path(cfg.RESULTS_DIR) / "benches" / f"{args.tag}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        run_bench_csv(m, n=args.n, oracle=False, outf=str(out_csv))

    elif args.cmd == "suite":
        # Full ICC battery or targeted slice
        from .bench_suite import (
            run_full_suite, run_all_benchmarks,
            B1_all_blind, B2_all_oracle, B3_by_K_blind, B4_by_SNR_blind,
            B5_by_range_bins_blind, B6_by_phi_fov_blind, B7_oracle_full_sweep,
            B4_k2_snr_sweep, B8_rmse_vs_K_at_snr, B9_heatmap_K_by_SNR,
        )

        if args.bench:
            # Run exactly one benchmark slice (nice for debugging figures)
            b = args.bench
            if b == "B1": B1_all_blind()
            elif b == "B2": B2_all_oracle()
            elif b == "B3": B3_by_K_blind()
            elif b == "B4": B4_by_SNR_blind()
            elif b == "B5": B5_by_range_bins_blind()
            elif b == "B6": B6_by_phi_fov_blind()
            elif b == "B7": B7_oracle_full_sweep()
            elif b == "B4_k2_snr_sweep": B4_k2_snr_sweep()
            elif b == "B8_rmse_vs_K_at_snr": B8_rmse_vs_K_at_snr(snr_target=args.snr, tol=args.tol)
            elif b == "B9_heatmap_K_by_SNR": B9_heatmap_K_by_SNR()
            else:
                raise SystemExit(f"Unknown bench {b}")
        else:
            # Full battery (B1..B7 + B4′ + B8 + B9)
            run_full_suite(include_decoupled=args.include_decoupled, limit=args.limit)
            if args.oracle_too:
                B2_all_oracle()


    elif args.cmd == "plots":
        try:
            # New: generates B1/B2 boxplots, K=2 SNR sweep, RMSE vs K @15dB, heatmaps
            from .plots_bench import make_all_plots
            make_all_plots()
        except ImportError:
            # Back-compat with your previous naming
            from .plots_bench import generate_all_plots
            generate_all_plots()

    elif args.cmd == "latex":
        try:
            from .tables import build_all_tables
            build_all_tables()
        except ImportError:
            from .tables import generate_all_tables
            generate_all_tables()

    elif args.cmd == "export":
        try:
            from .paper_export import export_paper_bundle
            export_paper_bundle()
        except ImportError:
            from .paper_export import export_paper_material
            export_paper_material()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

