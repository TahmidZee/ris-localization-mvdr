# Two-Stage HPO Strategy
**Date:** 2025-11-26  
**Status:** ✅ READY TO EXECUTE

---

## Overview

### Why Two Stages?

**Single-stage approach (old):**
- Run HPO with many trials
- Pick best trial
- Hope it generalizes to full data
- ⚠️ Problem: 10% subset may not perfectly predict full-data performance

**Two-stage approach (better):**
- **Stage 1:** Explore hyperparameter space on 10% data (fast, many trials)
- **Stage 2:** Refine top 5 configs on full 100K data (slow, few runs)
- ✅ Benefit: Best of both worlds - broad exploration + thorough validation

---

## Stage 1: HPO Exploration (10% Data)

### Goal
Find promising regions in hyperparameter space, not final hyperparameters.

### Configuration
```
Trials:           50 (40-60 recommended)
Epochs/trial:     20 max (most pruned at 6-10)
Early stopping:   6 epochs (aggressive)
Dataset:          10K train / 1K val (10% of full)
Time:             ~6-10 hours
```

### What Gets Explored
- Architecture: D_MODEL, NUM_HEADS, dropout
- Learning rate: log-uniform [1e-4, 4e-4]
- Loss weights: lam_cov, lam_ang, lam_rng
- Inference: range_grid, newton_iter, shrink_alpha
- Batch size: [64, 80]

### How to Run

**Option A: Bash (recommended)**
```bash
cd /home/tahit/ris/MainMusic
./run_hpo_manual.sh
```

**Option B: Python**
```bash
cd /home/tahit/ris/MainMusic
python run_hpo_manual.py
```

**Option C: Direct command**
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline hpo \
    --trials 50 \
    --hpo-epochs 20 \
    --space wide \
    --early-stop-patience 6
```

### Expected Output
```
[HPO Trial 0] Starting trial...
[HPO Trial 0] Training completed! Best objective: 0.456
...
[HPO Trial 49] Training completed! Best objective: 0.312
[HPO] best value=0.234 trial=27
```

### After Stage 1 Completes
- `results_final/hpo/best.json` - Best single config
- `results_final/hpo/hpo.db` - Full study database (for Stage 2)
- `results_final/hpo/*_trials.csv` - All trial results

---

## Stage 2: Refinement on Full Data

### Goal
Train top 5 configs on full dataset to find the true best model.

### Configuration
```
Configs to train:  5 (top 5 from Stage 1)
Epochs/run:        50 (full training)
Early stopping:    10 epochs (patient)
Dataset:           100K train / 10K val (full)
Time:              ~4-6 hours per run = ~20-30 hours total
```

### Why Top 5?
- Stage 1 ranking on 10% data isn't perfect
- The #3 config on 10% might be #1 on full data
- 5 runs is a good balance of coverage vs time

### How to Run (updated)

The legacy `run_stage2_refinement.*` scripts were removed during MVDR/K-head cleanup.

- **Backbone refinement (full data)**: run `python -m ris_pytorch_pipeline.ris_pipeline train ...` with your chosen HPO config(s).
- **SpectrumRefiner Stage-2 (Option B)**: after you have a good backbone checkpoint, train the refiner with:

```bash
python -m ris_pytorch_pipeline.ris_pipeline train-refiner \
  --backbone_ckpt results_final_L16_12x12/checkpoints/best.pt \
  --epochs 10 --use_shards --n_train 100000 --n_val 10000 \
  --lam_heatmap 0.1 --grid_phi 61 --grid_theta 41
```

### What Happens
1. Extracts top 5 configs from Stage 1 HPO database
2. For each config:
   - Creates dedicated output directory
   - Trains for 50 epochs on full 100K dataset
   - Applies early stopping (patience 10)
   - Saves best checkpoint
3. Compares final metrics across all 5 runs
4. Reports winner

### Expected Output (Stage-2 refiner)
```
[VAL] loss(first-batch)=...
Epoch ... train ... val ...
```

### After Stage 2 Completes
- `results_final/stage2/rank1_trial27/` - Best model checkpoint
- `results_final/stage2/top_configs.json` - All 5 configs
- `results_final/stage2/stage2_results.json` - Results summary
- `results_final/logs/stage2_*.log` - Training logs

---

## Complete Timeline

### Day 1 (Overnight)
```
6:00 PM  - Start Stage 1 HPO
          ./run_hpo_manual.sh

Next AM  - Stage 1 completes (~6-10 hours)
          Review results_final/hpo/best.json
```

### Day 2 (Overnight)
```
Morning  - Start Stage 2 Refinement
          python -m ris_pytorch_pipeline.ris_pipeline train --epochs 50 --use_shards --n_train 100000 --n_val 10000

Next AM  - Stage 2 completes (~20-30 hours)
          Compare metrics in results_final/stage2/
          Pick best model
```

### Day 3
```
Morning  - Tune thresholds on validation
          - MVDR_THRESH_DB: [8, 10, 12, 14]
          - HYBRID_COV_BETA: [0.0, 0.2, 0.3, 0.5]

Afternoon - Final test evaluation
          python -m ris_pytorch_pipeline.benchmark_test
          
Evening  - Write up results for paper!
```

---

## Quick Reference

### Stage 1 Commands
```bash
# Start Stage 1
./run_hpo_manual.sh

# Monitor progress
tail -f results_final/logs/hpo_stage1_*.log

# Check completed trials
python -c "
import optuna
s = optuna.load_study(
    study_name='L16_M64_wide_v2_optimfix',
    storage='sqlite:///results_final/hpo/hpo.db'
)
done = len([t for t in s.trials if t.state.is_finished()])
print(f'Completed: {done}/50')
"
```

### Stage 2 Commands
```bash
# Start Stage 2 (after Stage 1 completes)
python -m ris_pytorch_pipeline.ris_pipeline train --epochs 50 --use_shards --n_train 100000 --n_val 10000

# Optional: Stage 2b (MVDR SpectrumRefiner)
python -m ris_pytorch_pipeline.ris_pipeline train-refiner --backbone_ckpt <best.pt> --epochs 10 --use_shards

# Monitor progress
tail -f results_final/logs/*.log

# Check checkpoints
ls -la results_final_L16_12x12/checkpoints/
```

### After Both Stages
```bash
# View best Stage 1 config
cat results_final/hpo/best.json

# View Stage 2 comparison
cat results_final/stage2/stage2_results.json

# Pick winner and evaluate on test
python -m ris_pytorch_pipeline.benchmark_test \
    --checkpoint results_final/stage2/rank1_trial27/checkpoints/best.pt
```

---

## FAQ

### Q: Can I skip Stage 2 and just use best.json from Stage 1?
A: Yes, but you may not get the best model. Stage 1 ranking on 10% data isn't perfect.

### Q: What if Stage 1 gets interrupted?
A: Just run it again. Optuna resumes from the database.

### Q: What if Stage 2 gets interrupted?
A: You can manually restart individual configs. The script tracks which ones completed.

### Q: Can I run Stage 2 configs in parallel?
A: Yes! If you have multiple GPUs:
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python train_ris.py --from_hpo config_rank1.json --epochs 50 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python train_ris.py --from_hpo config_rank2.json --epochs 50 &
```

### Q: How do I know which Stage 2 model is best?
A: Compare these metrics:
1. **Angular/R range errors** (lower is better): e.g. φ/θ/r RMSE or medians
2. **Success_rate** (higher is better): % scenes with all sources within tolerance
3. **False positives**: detected sources / GT sources (closer to 1.0 is better)

Usually pick the model that minimizes errors while keeping success_rate high and false positives controlled.

### Q: What if all Stage 2 models have similar metrics?
A: Pick the one with:
- Lowest localization error (φ/θ/r RMSE/median)
- Highest success_rate
- Most stable training curve (check logs)
- Smallest checkpoint size (simpler model)

---

## Troubleshooting

### Stage 1 Issues

**"All trials get pruned"**
- Increase `n_warmup_steps` in hpo.py from 6 to 8-10
- Or reduce `early_stop_patience` from 6 to 4

**"Detection is too noisy / too many false positives"**
- Increase `MVDR_THRESH_DB` (more selective)
- Increase NMS separation / reduce `max_sources`
- Increase `HYBRID_COV_BETA` if shards have `R_samp`

**"CUDA out of memory"**
- Reduce batch_size range to [32, 64]
- Or disable gpu_cache

### Stage 2 Issues

**"Config extraction fails"**
- Verify Stage 1 completed: check `results_final/hpo/hpo.db`
- Check study name matches: "L16_M64_wide_v2_optimfix"

**"Training diverges"**
- Check learning rate isn't too high
- Verify shards are correct (100K samples)

**"Metrics don't improve from Stage 1"**
- This is normal! 10% subset can overfit
- Focus on robustness (SNR slices) and localization errors, not only surrogate score

---

## Summary

| Stage | Trials | Epochs | Dataset | Time | Goal |
|-------|--------|--------|---------|------|------|
| 1 | 50 | 20 max | 10K/1K | 6-10h | Find regions |
| 2 | 5 | 50 | 100K/10K | 20-30h | Find best model |

**Total time:** ~30-40 hours of computation

**Final output:** Production-ready model with:
- Tuned architecture and loss weights
- Validated on full dataset
- Ready for test evaluation

---

## Start Now!

```bash
cd /home/tahit/ris/MainMusic
./run_hpo_manual.sh
```

Then go to sleep 😴 and check results in the morning!



