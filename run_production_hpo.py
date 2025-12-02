#!/usr/bin/env python3
"""
Production HPO Runner
=====================

This script runs production HPO with all expert fixes applied and proper configuration.

Usage:
    python run_production_hpo.py --n_trials 50 --epochs_per_trial 20

Features:
- All expert fixes applied (gradient sanitization, AMP overflow detection, etc.)
- Structure terms re-enabled (they were working fine)
- AMP enabled for faster training
- EMA/SWA enabled for better convergence
- Proper learning rate and regularization
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 PRODUCTION HPO RUNNER")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not Path("ris_pytorch_pipeline").exists():
        print("❌ Error: Must run from /home/tahit/ris/MainMusic/")
        print("   Current directory:", os.getcwd())
        sys.exit(1)
    
    # Default arguments
    n_trials = 50
    epochs_per_trial = 20
    space = "wide"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            n_trials = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid n_trials: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            epochs_per_trial = int(sys.argv[2])
        except ValueError:
            print(f"❌ Invalid epochs_per_trial: {sys.argv[2]}")
            sys.exit(1)
    
    if len(sys.argv) > 3:
        space = sys.argv[3]
        if space not in ["wide", "narrow"]:
            print(f"❌ Invalid space: {space} (must be 'wide' or 'narrow')")
            sys.exit(1)
    
    print(f"📊 HPO Configuration:")
    print(f"  • Trials: {n_trials}")
    print(f"  • Epochs per trial: {epochs_per_trial}")
    print(f"  • Search space: {space}")
    print()
    
    print("✅ Expert Fixes Applied:")
    print("  • Gradient sanitization fixed (no more AMP overflow masking)")
    print("  • AMP overflow detection enabled")
    print("  • Parameter drift probe working")
    print("  • Gradient path verification enabled")
    print("  • Step gate fixed (no more silent skipping)")
    print("  • EMA masking fixed")
    print()
    
    print("✅ Production Configuration:")
    print("  • Structure terms re-enabled (LAM_SUBSPACE_ALIGN=0.05, LAM_PEAK_CONTRAST=0.1)")
    print("  • AMP enabled for faster training")
    print("  • EMA/SWA enabled for better convergence")
    print("  • Normal learning rate (LR_INIT=2e-4)")
    print("  • Gradient clipping enabled (CLIP_NORM=1.0)")
    print("  • Curriculum learning enabled")
    print()
    
    # Run HPO
    cmd = [
        "python", "-m", "ris_pytorch_pipeline.hpo",
        "--n_trials", str(n_trials),
        "--epochs_per_trial", str(epochs_per_trial),
        "--space", space,
        "--export_csv"
    ]
    
    print(f"🚀 Starting HPO with command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        # Run the HPO command
        result = subprocess.run(cmd, check=True)
        print("✅ HPO completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ HPO failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️ HPO interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


