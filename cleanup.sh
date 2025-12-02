#!/bin/bash
# Production Cleanup Script
# Removes old documentation, scripts, and backup files

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================="
echo "Production Cleanup Script"
echo "========================================="
echo ""

# Confirm with user
read -p "This will remove old documentation, scripts, and backups. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cleanup cancelled."
    exit 1
fi

echo ""
echo "Step 1: Removing old documentation..."
rm -f ALL_FIXES_COMPLETE.md ALL_INTEGRATION_DONE.md ALL_ITEMS_ADDRESSED.md \
      BASELINE_L100_PIPELINE.md BASELINE_SETUP_SUMMARY.md CODEBASE_STRUCTURE.md \
      CRITICAL_DIAGNOSIS.md DIAGNOSIS_SUMMARY.md FINAL_ACTION_PLAN.md \
      FINAL_SUMMARY.md GAP_ANALYSIS_FINAL.md IMPLEMENTATION_SUMMARY.md \
      INFERENCE_ISSUE_DIAGNOSIS.md INTEGRATION_COMPLETE.md PATH_TO_SUB_1_DEGREE.md \
      PRODUCTION_READINESS_CHECKLIST.md QUICK_START.md READY_TO_RETRAIN.md \
      REFINED_PATH_TO_SUB_1_DEGREE.md SLURM_TRAINING_GUIDE.md CLEANUP_AND_RUN.md

echo "✓ Old documentation removed"

echo ""
echo "Step 2: Removing old test scripts..."
rm -f test_immediate_fixes.py test_raw_model.py test_raw_only.py test_theta_convention.py \
      simple_benchmark.py quick_diagnosis.py proper_benchmark.py \
      benchmark_raw_output.py benchmark_l16_final.py compare_raw_vs_refined.py \
      comprehensive_benchmark_L8.py run_baselines_bench.py run_ours_comprehensive.py

echo "✓ Old test scripts removed"

echo ""
echo "Step 3: Removing old conversion scripts..."
rm -f convert_L100_*.py convert_L8_*.py

echo "✓ Old conversion scripts removed"

echo ""
echo "Step 4: Removing old training scripts..."
rm -f train_all_L16_12x12_models.sh train_baselines_L100*.py \
      train_dcd_*.py train_L16_12x12_final.sh train_nfssn_cluster.py

echo "✓ Old training scripts removed"

echo ""
echo "Step 5: Removing old pipeline scripts..."
rm -f backup_L8_data.sh benchmark_L16_12x12.sh full_L16_12x12_pipeline.sh \
      generate_L16_12x12_data.sh run_L16_12x12_hpo.sh

echo "✓ Old pipeline scripts removed"

echo ""
echo "Step 6: Removing SLURM scripts..."
rm -f slurm_*.sh

echo "✓ SLURM scripts removed"

echo ""
echo "Step 7: Removing duplicate/old code files..."
rm -f ris_pytorch_pipeline/configs_fixed.py \
      ris_pytorch_pipeline/infer_fixed.py \
      ris_pytorch_pipeline/infer_old_backup.py \
      ris_pytorch_pipeline/physics_fixed.py

echo "✓ Duplicate code files removed"

echo ""
echo "Step 8: Archiving old experiments..."
mkdir -p archive_old_experiments
[ -d "results_final_L8" ] && mv results_final_L8 archive_old_experiments/ 2>/dev/null || true
[ -d "results_baselines_L100_12x12" ] && mv results_baselines_L100_12x12 archive_old_experiments/ 2>/dev/null || true

echo "✓ Old results archived to archive_old_experiments/"

echo ""
echo "Step 9: Removing old backup directories..."
rm -rf backup_L4_system/ backup_L8_20250929_181218/ OldCode/

echo "✓ Old backups removed"

echo ""
echo "========================================="
echo "Cleanup Complete!"
echo "========================================="
echo ""
echo "Production files remaining:"
echo "  - ris_pytorch_pipeline/        (main codebase)"
echo "  - stratified_evaluation.py     (evaluation tool)"
echo "  - 1_regenerate_data.sh         (data generation)"
echo "  - 2_train_model.sh             (training)"
echo "  - 3_evaluate_model.sh          (evaluation)"
echo "  - README_PRODUCTION.md         (documentation)"
echo "  - CODEBASE_AUDIT_FINDINGS.md   (audit report)"
echo ""
echo "Archived:"
echo "  - archive_old_experiments/     (old results)"
echo ""

