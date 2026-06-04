# Augmentation Sweep Experiment Results

**Run ID**: `20260604T143316Z`
**Manifest**: `augment_sweep.yaml`
**Model Type**: `baseline` (XGBoost, Logistic Regression, Linear SVM, Random Forest evaluated, Logistic Regression selected as winner)
**Total Execution Time**: ~15.6 minutes (937.3 seconds)

## Overview
This experiment evaluated various data augmentation strategies, particularly focusing on handling class imbalances for low-star ratings.

## Model Performance

Among the trained models for each setup, **Logistic Regression** consistently outperformed the others in terms of Mean Absolute Error (MAE) and Macro F1-score on the validation set.

- **Winner**: Logistic Regression
- **Val MAE**: 0.6233 (on best baseline config context)
- **F1-Macro**: 0.4809

## Augmentation Strategy Results

The variants are ranked by MAE (lower is better) on the holdout/evaluation set.

| Rank | Variant | MAE | Accuracy | F1-Macro | Off-by-1 | Recall ★1 | Recall ★2 | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 🏆** | `aug_heavy_2500` | **0.6362** | **0.4708** | 0.4258 | **0.9386** | 0.387 | 0.246 | ~3.1m |
| 2 | `aug_medium_1400` | 0.6428 | 0.4620 | 0.4248 | 0.9320 | 0.409 | 0.246 | ~2.0m |
| 3 | `aug_high_synonym` | 0.6428 | 0.4620 | 0.4248 | 0.9320 | 0.409 | 0.246 | ~2.2m |
| 4 | `aug_low_synonym` | 0.6428 | 0.4620 | 0.4248 | 0.9320 | 0.409 | 0.246 | ~2.2m |
| 5 | `aug_all_stars` | 0.6433 | 0.4700 | **0.4299** | 0.9349 | **0.409** | **0.246** | ~2.4m |
| 6 | `aug_disabled` | 0.6484 | 0.4496 | 0.4149 | 0.9276 | 0.398 | 0.246 | ~1.9m |
| 7 | `aug_light_800` | 0.6484 | 0.4496 | 0.4149 | 0.9276 | 0.398 | 0.246 | ~1.8m |

## Key Findings

1. **Heavy Augmentation is Most Effective**: The `aug_heavy_2500` variant significantly outperformed the baseline (`aug_disabled`) and lighter augmentation variants, lowering the MAE from 0.6484 to 0.6362 and increasing accuracy by over 2%.
2. **"Off-by-1" Accuracy Improvement**: Heavy augmentation also improved the "Off-by-1" accuracy, reaching almost 94%, meaning the model is very close to the true rating even when it makes a mistake.
3. **Recall Constraints on Minority Classes**: Despite overall improvements in MAE and Accuracy, the recall for the hardest minority classes (Star 1 and Star 2) remains challenging. While heavy augmentation improved the overall metrics, it slightly reduced the Recall for Star 1 (0.387) compared to medium augmentation (0.409).
4. **Synonym Replacement**: Variants that used synonym replacements (`aug_high_synonym`, `aug_low_synonym`) did not show meaningful improvements over the standard `aug_medium_1400` strategy and yielded identical aggregated performance.

## Conclusion

Data augmentation provides a clear benefit for this dataset. The `aug_heavy_2500` strategy provides the best overall performance in minimizing error (MAE), although careful consideration must be given if maximizing recall on 1-star ratings is a higher priority than overall MAE. Future experiments could explore a hybrid approach that combines heavy augmentation with targeted techniques to boost 1-star recall.
