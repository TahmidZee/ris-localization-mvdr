#!/usr/bin/env python3
"""
STAGE 2: REFINEMENT ON FULL DATA
================================
Train top 5 configs from Stage 1 HPO on full 100K dataset
Date: 2025-11-26

PREREQUISITES:
  - Stage 1 HPO completed (./run_hpo_manual.sh or run_hpo_manual.py)
  - results_final/hpo/hpo.db exists with completed trials

WHAT THIS DOES:
  - Extract top 5 configs by K_loc score from Stage 1
  - Train each on full dataset (100K train / 10K val)
  - 50 epochs with early stopping (patience 8-10)
  - Save best model for each config
  - Compare final metrics and pick winner
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 80)
    print("STAGE 2: REFINEMENT ON FULL DATA")
    print("=" * 80)
    print()
    
    # Configuration
    TOP_K = 5              # Number of top configs to refine
    EPOCHS = 50            # Full training epochs
    EARLY_STOP = 10        # Early stopping patience (longer than Stage 1)
    BATCH_SIZE = 64        # Batch size for full training
    
    # Paths
    PROJECT_DIR = Path(__file__).parent
    HPO_DIR = PROJECT_DIR / "results_final" / "hpo"
    STAGE2_DIR = PROJECT_DIR / "results_final" / "stage2"
    LOG_DIR = PROJECT_DIR / "results_final" / "logs"
    
    print("Configuration:")
    print(f"  Top configs to refine: {TOP_K}")
    print(f"  Epochs per run: {EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOP} epochs")
    print(f"  Batch size: {BATCH_SIZE}")
    print()
    
    # Verify Stage 1 completed
    hpo_db = HPO_DIR / "hpo.db"
    if not hpo_db.exists():
        print("✗ ERROR: Stage 1 HPO database not found!")
        print(f"  Expected: {hpo_db}")
        print("  Please run ./run_hpo_manual.sh first")
        sys.exit(1)
    print("✓ Stage 1 HPO database found")
    
    # Create output directories
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract top K configs
    print()
    print(f"Extracting top {TOP_K} configs from Stage 1...")
    
    try:
        import optuna
        
        study = optuna.load_study(
            study_name="L16_M64_wide_v2_optimfix",
            storage=f"sqlite:///{hpo_db}"
        )
        
        # Get completed trials sorted by value (lower is better)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed.sort(key=lambda t: t.value)
        
        # Take top K
        top_trials = completed[:TOP_K]
        
        # Format configs
        configs = []
        for i, trial in enumerate(top_trials):
            configs.append({
                "rank": i + 1,
                "trial_number": trial.number,
                "objective": trial.value,
                "params": trial.params
            })
        
        # Save to file
        configs_file = STAGE2_DIR / "top_configs.json"
        with open(configs_file, 'w') as f:
            json.dump(configs, f, indent=2)
        
        print(f"✓ Top {TOP_K} configs extracted to: {configs_file}")
        print()
        print(json.dumps(configs, indent=2))
        print()
        
    except Exception as e:
        print(f"✗ ERROR: Failed to extract top configs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)
    print("Starting Stage 2 Refinement Training")
    print("=" * 80)
    print()
    print(f"This will train {TOP_K} models on full dataset (100K samples)")
    print("Expected time: ~4-6 hours per model = ~20-30 hours total")
    print()
    print("Press Ctrl+C within 5 seconds to abort...")
    
    import time
    for i in range(5, 0, -1):
        print(f"  Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print()
    print()
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Results tracking
    results = []
    
    # Train each top config
    for config in configs:
        rank = config["rank"]
        trial_num = config["trial_number"]
        params = config["params"]
        
        print()
        print("=" * 70)
        print(f"Training Config #{rank} of {TOP_K} (Trial #{trial_num})")
        print("=" * 70)
        print()
        print(f"Parameters: {json.dumps(params, indent=2)}")
        print()
        
        # Create config file for this run
        config_file = STAGE2_DIR / f"config_rank{rank}.json"
        with open(config_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Output directory for this run
        run_dir = STAGE2_DIR / f"rank{rank}_trial{trial_num}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = LOG_DIR / f"stage2_rank{rank}_{timestamp}.log"
        
        print(f"Output: {run_dir}")
        print(f"Log: {log_file}")
        print()
        
        # Build command
        cmd = [
            sys.executable, "-m", "ris_pytorch_pipeline.ris_pipeline", "train",
            "--epochs", str(EPOCHS),
            "--use_shards",
            "--from_hpo", str(config_file),
            "--batch-size", str(BATCH_SIZE),
            "--early-stop-patience", str(EARLY_STOP),
            "--output-dir", str(run_dir),
        ]
        
        # Run training
        try:
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=PROJECT_DIR
                )
                
                # Stream output to both console and log file
                for line in process.stdout:
                    print(line, end='')
                    log_f.write(line)
                    log_f.flush()
                
                process.wait()
                exit_code = process.returncode
            
            if exit_code == 0:
                print(f"✓ Config #{rank} training completed")
                results.append({
                    "rank": rank,
                    "trial_number": trial_num,
                    "status": "success",
                    "run_dir": str(run_dir)
                })
            else:
                print(f"⚠️  Config #{rank} training failed (exit code: {exit_code})")
                results.append({
                    "rank": rank,
                    "trial_number": trial_num,
                    "status": "failed",
                    "exit_code": exit_code,
                    "run_dir": str(run_dir)
                })
                
        except KeyboardInterrupt:
            print()
            print(f"⚠️  Config #{rank} interrupted by user")
            results.append({
                "rank": rank,
                "trial_number": trial_num,
                "status": "interrupted",
                "run_dir": str(run_dir)
            })
            break
            
        except Exception as e:
            print(f"⚠️  Config #{rank} training error: {e}")
            results.append({
                "rank": rank,
                "trial_number": trial_num,
                "status": "error",
                "error": str(e),
                "run_dir": str(run_dir)
            })
    
    # Save results summary
    results_file = STAGE2_DIR / "stage2_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 80)
    print("STAGE 2 REFINEMENT COMPLETE")
    print("=" * 80)
    print()
    
    # Display results comparison
    print("=" * 70)
    print("STAGE 2 RESULTS COMPARISON")
    print("=" * 70)
    print()
    
    for r in results:
        print(f"Rank #{r['rank']} (Trial #{r['trial_number']}): {r['status']}")
        if r['status'] == 'success':
            # Try to load metrics
            run_dir = Path(r['run_dir'])
            metrics_file = run_dir / "final_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                print(f"  K_acc: {metrics.get('K_acc', 'N/A')}")
                print(f"  K_under: {metrics.get('K_under', 'N/A')}")
                print(f"  AoA_RMSE: {metrics.get('AoA_RMSE', 'N/A')}")
                print(f"  Range_RMSE: {metrics.get('Range_RMSE', 'N/A')}")
                print(f"  Success_rate: {metrics.get('Success_rate', 'N/A')}")
            else:
                print(f"  Metrics: Check {run_dir}/checkpoints/ or log file")
        elif r['status'] == 'failed':
            print(f"  Exit code: {r.get('exit_code', 'unknown')}")
        elif r['status'] == 'error':
            print(f"  Error: {r.get('error', 'unknown')}")
        print()
    
    print("=" * 70)
    print("RECOMMENDATION:")
    print("  1. Check each run's log for final validation metrics")
    print("  2. Pick the model with highest K_acc and lowest K_under")
    print("  3. Run final test evaluation:")
    print("     python -m ris_pytorch_pipeline.benchmark_test --checkpoint <best_model.pt>")
    print("=" * 70)
    print()
    print(f"Results summary: {results_file}")
    print(f"Logs: {LOG_DIR}/stage2_*.log")


if __name__ == "__main__":
    main()



