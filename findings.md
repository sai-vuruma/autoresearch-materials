## Current best
val_mae: 9.627624 (baseline, commit 6acd322)
Key config: GPRegressor, StandardScaler on X, Yeo-Johnson target transform, ConstantKernel * Matern(nu=1.5) + WhiteKernel, alpha=1e-4, n_restarts_optimizer=2.

## What works
- Baseline Matern GP with target power transform is the best so far: val_mae 9.627624.

## What doesn't work
- DotProduct + RBF kernel replacing the baseline Matern kernel regressed badly: val_mae 13.050588.
- Adding DotProduct to the baseline Matern kernel was slightly worse: val_mae 9.658747.
- Removing the Yeo-Johnson target transform was slightly worse: val_mae 9.662166.
- Increasing GP optimizer restarts from 2 to 8 did not improve: val_mae 9.627625.
- Smoother Matern nu=2.5 regressed: val_mae 9.983366.

## Structural findings
- The validation split appears sensitive to GP tail behavior; explicit additive linear kernel terms have not improved extrapolation despite guidance suggesting linear+local GP kernels as a strong general OOD baseline.
- The target power transform is beneficial or at least not removable for this current GP.

## Unexplored directions
- Try rougher Matern nu=0.5 or RationalQuadratic local kernels.
- Try delta-learning: Ridge extrapolating baseline plus GP or residual model.
- Try small ensembles/blends of Ridge, GP, and tree/boosting models if available in scikit-learn.
- Try target quantile or shifted/asymmetric residual adjustments aimed at high-y OOD underprediction.
