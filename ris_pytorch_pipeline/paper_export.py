# SPDX-License-Identifier: MIT
import shutil
from pathlib import Path

ROOT = Path(".")
FIG_DIR = ROOT/"results_final/figures"
TAB_DIR = ROOT/"results_final/tables"
EXP_DIR = ROOT/"results_final/paper_export"

def export_paper_bundle():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    # copy figures
    if FIG_DIR.exists():
        for p in FIG_DIR.glob("*.png"):
            shutil.copy2(p, EXP_DIR/p.name)
    # copy tables
    if TAB_DIR.exists():
        for p in TAB_DIR.glob("*.csv"):
            shutil.copy2(p, EXP_DIR/p.name)
    print("Exported to", EXP_DIR)
