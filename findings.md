## Current best
val_mae: 7.778742 (commit e701362)
Key config: BiasShiftedGPRegressor, StandardScaler on X, Yeo-Johnson target transform, ConstantKernel * Matern(nu=1.5) + WhiteKernel, alpha=1e-4, n_restarts_optimizer=2; bias estimated from top-y 20% training holdout after fitting calibration GP on lower-y 80%.

## What works
- Baseline Matern GP with target power transform is the best so far: val_mae 9.627624.
- Bias-only calibration from a train-only high-y holdout substantially improves OOD validation: val_mae 7.778742. Internal diagnostic showed baseline GP underpredicted top-y training holdout by about 5.26 on average; applying that mean shift generalized better than affine calibration.

## What doesn't work
- DotProduct + RBF kernel replacing the baseline Matern kernel regressed badly: val_mae 13.050588.
- Adding DotProduct to the baseline Matern kernel was slightly worse: val_mae 9.658747.
- Removing the Yeo-Johnson target transform was slightly worse: val_mae 9.662166.
- Increasing GP optimizer restarts from 2 to 8 did not improve: val_mae 9.627625.
- Smoother Matern nu=2.5 regressed: val_mae 9.983366.
- Rougher Matern nu=0.5 regressed badly: val_mae 12.573577.
- RationalQuadratic local kernel regressed: val_mae 10.766421.
- Plain RidgeCV was much worse: val_mae 18.957571.
- RidgeCV plus baseline GP residual correction was slightly worse than baseline: val_mae 9.684419.
- Full affine calibration on high-y holdout overcorrected badly: val_mae 15.288201.

## Structural findings
- The validation split appears sensitive to GP tail behavior; explicit additive linear kernel terms have not improved extrapolation despite guidance suggesting linear+local GP kernels as a strong general OOD baseline.
- The target power transform is beneficial or at least not removable for this current GP.
- The baseline GP underpredicts high-y OOD regions. A constant bias correction learned from top-y train holdout is robust; slope/intercept correction is too aggressive.

## Unexplored directions
- Tune the bias calibration split fraction and shrinkage multiplier.
- Try bias correction estimated from multiple high-y holdout folds and average the shifts.
- Try small ensembles/blends of Ridge, GP, and tree/boosting models if available in scikit-learn.
- Try target quantile or shifted/asymmetric residual adjustments aimed at high-y OOD underprediction.
