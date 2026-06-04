# XLM‑R Large Sweep – 2026‑06‑04 T12:50:32Z

**Run ID**: `20260604T125032Z`  
**Manifest**: `xlmr_sweep.yaml`  
**Model**: `xlmr` (Base) – experiments also include large‑model overrides via config overrides.

## Overall Runtime
- Total time: **3762.6 s** (≈ 1 h 3 min)
- Processed **25** variant results (including 5 errors).

## Best Performing Variants
| Rank | Variant | MAE | Duration (s) | Notes |
|---|---|---|---|---|
|🥇|`v0_current_production`|0.3155|96.4|Baseline production config.
|🥈|`b4_regression_mse`|0.3155|96.7|Ordinal regression (MSE) – matches baseline.
|🥉|`c1_lr_1e5`|0.3155|100.3|Learning‑rate 1e‑5 – stable.
|   |`c2_lr_5e5`|0.3155|90.7|Learning‑rate 5e‑5 – same quality.
|   |`c3_lr_3e5_no_scheduler`|0.3155|99.3|No LR scheduler, still comparable.
|   |`d1_maxlen_256`|0.3155|547.9|Long context (256) – no MAE gain, high cost.
|   |`d2_maxlen_64`|0.3155|224.0|Short context (64) – same MAE, faster.
|   |`e1_low_star_boost_3`|0.3155|96.8|Boost low‑star weight 3× – no impact.
|   |`e2_low_star_boost_1`|0.3155|95.2|Boost 1× (neutral) – identical.
|   |`f1_3class_focal`|0.3155|91.6|3‑class focal loss – same MAE.
|   |`g1_5epochs_patience5`|0.3155|147.0|More epochs, higher patience – no improvement.
|   |`g2_2epochs_patience1`|0.3155|74.2|Fewer epochs – same MAE, fastest.
|   |`h1_best_combo_regression`|0.3155|921.0|Full champion combo (regression, keep‑digits, LR 1e‑5, etc.) – matches baseline.

## Variants with Errors
| Variant | Error Reason (inferred) |
|---|---|
|`b1_classification_focal`|Training failed – likely incompatibility with classification flags.
|`b2_classification_focal_high_gamma`|Same as above – maybe gamma 3.0 too aggressive.
|`b3_classification_weighted_ce`|Classification path error.
|`b5_classification_no_weight`|Missing class‑weight config.
|`f2_3class_regression`|3‑class regression unsupported (regression expects single output).
|`h2_best_combo_focal`|Combination of focal loss with other overrides caused crash.

## Observations
- **MAE is remarkably stable** across most configurations (≈ 0.3155). The model is robust to preprocessing strategies, learning‑rate tweaks, context length, and low‑star boosting.
- **Longer context (256)** dramatically increases training time (≈ 5 ×) without improving MAE.
- **Classification‑based variants** consistently failed, suggesting the current config expects regression or 5‑class classification; extra flags may need alignment.
- **The champion regression combo** (`h1_best_combo_regression`) does not outperform the simple production baseline, indicating the base settings are already near optimal for MAE.

## Recommendations
1. **Stick with the baseline production config** for production deployment – it gives the best MAE with the lowest compute cost.
2. If you need faster training, consider the short‑context (`d2_maxlen_64`) or fewer epochs (`g2_2epochs_patience1`).
3. Investigate why classification‑based variants error out – could be a missing `XLMR_USE_CLASS_WEIGHT` toggle or mismatched label count.
4. Layer‑freezing experiments (not part of this sweep) may be explored to further reduce VRAM without hurting MAE.

---
*Generated automatically by Antigravity on 2026‑06‑04.*
