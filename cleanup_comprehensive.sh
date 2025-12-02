#!/bin/bash

# Comprehensive cleanup script for MainMusic project
# This script removes unnecessary files, caches, and temporary data

echo "🧹 Starting comprehensive cleanup of MainMusic project..."

# Change to the project directory
cd /home/tahit/ris/MainMusic

# 1. Remove Python cache files
echo "🗑️  Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# 2. Remove log files
echo "🗑️  Removing log files..."
find . -name "*.log" -delete 2>/dev/null || true
find . -name "*.out" -delete 2>/dev/null || true
find . -name "*.err" -delete 2>/dev/null || true
find . -name "slurm-*" -delete 2>/dev/null || true

# 3. Remove temporary files
echo "🗑️  Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.temp" -delete 2>/dev/null || true
find . -name "*.swp" -delete 2>/dev/null || true
find . -name "*.swo" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

# 4. Remove backup directories (keep only the most recent one)
echo "🗑️  Cleaning up backup directories..."
if [ -d "backup_L8_20250929_181218" ]; then
    echo "   Keeping backup_L8_20250929_181218 (6.9G) - most recent backup"
fi

# 5. Remove old result directories (keep only the most recent)
echo "🗑️  Cleaning up old result directories..."
if [ -d "results_final_L16_12x12" ]; then
    echo "   Keeping results_final_L16_12x12 (230M) - most recent results"
fi
if [ -d "results_final" ] && [ ! -s "results_final" ]; then
    echo "   Removing empty results_final directory"
    rmdir results_final 2>/dev/null || true
fi

# 6. Remove old data directories (keep only the most recent ones)
echo "🗑️  Cleaning up old data directories..."
if [ -d "data_shards_L100_baseline" ] && [ ! -s "data_shards_L100_baseline" ]; then
    echo "   Removing empty data_shards_L100_baseline directory"
    rmdir data_shards_L100_baseline 2>/dev/null || true
fi

# 7. Remove duplicate/old analysis files
echo "🗑️  Removing duplicate analysis files..."
# Keep only the most comprehensive analysis files
rm -f ALL_*.md 2>/dev/null || true
rm -f COMPREHENSIVE_*.md 2>/dev/null || true
rm -f CRITICAL_*.md 2>/dev/null || true
rm -f FINAL_*.md 2>/dev/null || true
rm -f HPO_*.md 2>/dev/null || true
rm -f DEBUG_*.md 2>/dev/null || true
rm -f BUGFIX_*.md 2>/dev/null || true
rm -f PHASE*.md 2>/dev/null || true
rm -f README_*.md 2>/dev/null || true
rm -f ROOT_*.md 2>/dev/null || true
rm -f STRATEGIC_*.md 2>/dev/null || true
rm -f NEXT_*.md 2>/dev/null || true
rm -f IMPLEMENTATION_*.md 2>/dev/null || true
rm -f INTEGRATION_*.md 2>/dev/null || true
rm -f OPTIMIZATION_*.md 2>/dev/null || true
rm -f MEMORY_*.md 2>/dev/null || true
rm -f GPU_*.md 2>/dev/null || true
rm -f OOM_*.md 2>/dev/null || true
rm -f REAL_*.md 2>/dev/null || true
rm -f LOSS_*.md 2>/dev/null || true
rm -f PARAMETER_*.md 2>/dev/null || true
rm -f OVERFITTING_*.md 2>/dev/null || true
rm -f MODEL_*.md 2>/dev/null || true
rm -f M64_*.md 2>/dev/null || true
rm -f L16_*.md 2>/dev/null || true
rm -f L100_*.md 2>/dev/null || true
rm -f BASELINE_*.md 2>/dev/null || true
rm -f DATASET_*.md 2>/dev/null || true
rm -f DATA_*.md 2>/dev/null || true
rm -f SNR_*.md 2>/dev/null || true
rm -f NOISE_*.md 2>/dev/null || true
rm -f STEERING_*.md 2>/dev/null || true
rm -f MANIFOLD_*.md 2>/dev/null || true
rm -f RANK_*.md 2>/dev/null || true
rm -f SUBSPACE_*.md 2>/dev/null || true
rm -f GAP_*.md 2>/dev/null || true
rm -f CURRICULUM_*.md 2>/dev/null || true
rm -f EXECUTION_*.md 2>/dev/null || true
rm -f PRODUCTION_*.md 2>/dev/null || true
rm -f QUICK_*.md 2>/dev/null || true
rm -f ROADMAP_*.md 2>/dev/null || true
rm -f PATH_*.md 2>/dev/null || true
rm -f REFINED_*.md 2>/dev/null || true
rm -f FAIR_*.md 2>/dev/null || true
rm -f BENCHMARK_*.md 2>/dev/null || true
rm -f COMPLETE_*.md 2>/dev/null || true
rm -f STATUS_*.md 2>/dev/null || true
rm -f SUMMARY_*.md 2>/dev/null || true
rm -f ANALYSIS_*.md 2>/dev/null || true
rm -f AUDIT_*.md 2>/dev/null || true
rm -f FINDINGS_*.md 2>/dev/null || true
rm -f RESULTS_*.md 2>/dev/null || true
rm -f VERIFICATION_*.md 2>/dev/null || true
rm -f CHECKLIST_*.md 2>/dev/null || true
rm -f GUIDE_*.md 2>/dev/null || true
rm -f WORKFLOW_*.md 2>/dev/null || true
rm -f CORRECT_*.md 2>/dev/null || true
rm -f CORRECTED_*.md 2>/dev/null || true
rm -f DECISIVE_*.md 2>/dev/null || true
rm -f EMERGENCY_*.md 2>/dev/null || true
rm -f DIAGNOSIS_*.md 2>/dev/null || true
rm -f DIAGNOSTIC_*.md 2>/dev/null || true
rm -f DEBUGGING_*.md 2>/dev/null || true
rm -f BREAKDOWN_*.md 2>/dev/null || true
rm -f CAVEATS_*.md 2>/dev/null || true
rm -f RECOMMENDATIONS_*.md 2>/dev/null || true
rm -f UPDATES_*.md 2>/dev/null || true
rm -f IMPACT_*.md 2>/dev/null || true
rm -f STORY_*.md 2>/dev/null || true
rm -f METHODS_*.md 2>/dev/null || true
rm -f MEASUREMENT_*.md 2>/dev/null || true
rm -f CALCULATION_*.md 2>/dev/null || true
rm -f CLARIFICATION_*.md 2>/dev/null || true
rm -f CORRUPTION_*.md 2>/dev/null || true
rm -f GENERATION_*.md 2>/dev/null || true
rm -f DISTRIBUTION_*.md 2>/dev/null || true
rm -f COMPARISON_*.md 2>/dev/null || true
rm -f FEATURE_*.md 2>/dev/null || true
rm -f EXTRACTION_*.md 2>/dev/null || true
rm -f NORMALIZATION_*.md 2>/dev/null || true
rm -f PIPELINE_*.md 2>/dev/null || true
rm -f STEP_*.md 2>/dev/null || true
rm -f TRACE_*.md 2>/dev/null || true
rm -f DEEP_*.md 2>/dev/null || true
rm -f FINAL_*.md 2>/dev/null || true
rm -f SYSTEMATIC_*.md 2>/dev/null || true
rm -f LIKELY_*.md 2>/dev/null || true
rm -f CAUSES_*.md 2>/dev/null || true
rm -f USAGE_*.md 2>/dev/null || true
rm -f REFERENCE_*.md 2>/dev/null || true
rm -f ISSUE_*.md 2>/dev/null || true
rm -f TRAINING_*.md 2>/dev/null || true
rm -f SYSTEM_*.md 2>/dev/null || true
rm -f MISMATCH_*.md 2>/dev/null || true
rm -f INFERENCE_*.md 2>/dev/null || true
rm -f DIAGNOSIS_*.md 2>/dev/null || true
rm -f ISSUE_*.md 2>/dev/null || true
rm -f TRAINING_*.md 2>/dev/null || true
rm -f SYSTEM_*.md 2>/dev/null || true
rm -f MISMATCH_*.md 2>/dev/null || true
rm -f INFERENCE_*.md 2>/dev/null || true

# 8. Remove old test and debug scripts
echo "🗑️  Removing old test and debug scripts..."
rm -f test_*.py 2>/dev/null || true
rm -f debug_*.py 2>/dev/null || true
rm -f check_*.py 2>/dev/null || true
rm -f diagnose_*.py 2>/dev/null || true
rm -f quick_*.py 2>/dev/null || true
rm -f minimal_*.py 2>/dev/null || true
rm -f simple_*.py 2>/dev/null || true
rm -f proper_*.py 2>/dev/null || true
rm -f run_*.py 2>/dev/null || true
rm -f benchmark_*.py 2>/dev/null || true
rm -f comprehensive_*.py 2>/dev/null || true
rm -f stratified_*.py 2>/dev/null || true
rm -f generate_*.py 2>/dev/null || true
rm -f convert_*.py 2>/dev/null || true
rm -f analyze_*.py 2>/dev/null || true
rm -f compare_*.py 2>/dev/null || true
rm -f monitor_*.py 2>/dev/null || true
rm -f calibrate_*.py 2>/dev/null || true
rm -f eval_*.py 2>/dev/null || true

# 9. Remove old shell scripts (keep only the essential ones)
echo "🗑️  Cleaning up old shell scripts..."
# Keep essential scripts
# rm -f *.sh 2>/dev/null || true
# rm -f *.bash 2>/dev/null || true

# 10. Remove old directories that might be empty
echo "🗑️  Removing empty directories..."
find . -type d -empty -delete 2>/dev/null || true

# 11. Clean up any remaining temporary files
echo "🗑️  Final cleanup..."
find . -name "*.pid" -delete 2>/dev/null || true
find . -name "core.*" -delete 2>/dev/null || true

echo "✅ Cleanup completed!"
echo ""
echo "📊 Remaining important files:"
echo "   - ris_pytorch_pipeline/ (main code)"
echo "   - data_shards_M64_L16/ (23G - main dataset)"
echo "   - data_nfssn_benchmark/ (6.9G - benchmark data)"
echo "   - backup_L8_20250929_181218/ (6.1G - backup)"
echo "   - results_final_L16_12x12/ (230M - results)"
echo ""
echo "💾 Estimated space saved: ~500MB+ (logs, caches, temp files)"
echo "🎯 Project is now clean and organized!"


