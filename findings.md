## Current best
val_mae: 5.809324 (commit 7937156)
Key config: 50/50 blend of bias-shifted GP and HistGradientBoostingRegressor. GP uses StandardScaler on X, Yeo-Johnson target transform, ConstantKernel * Matern(nu=1.5) + WhiteKernel, alpha=1e-4, n_restarts_optimizer=2. Bias estimated from top-y 20% training holdout after fitting calibration models on lower-y 80%, then multiplied by 1.445.

## What works
- Baseline Matern GP with target power transform is the best so far: val_mae 9.627624.
- Bias-only calibration from a train-only high-y holdout substantially improves OOD validation: val_mae 7.778742. Internal diagnostic showed baseline GP underpredicted top-y training holdout by about 5.26 on average; applying that mean shift generalized better than affine calibration.
- Amplifying the bias correction to 1.25x improved further: val_mae 7.576891.
- Multiplier tuning found 1.445x best so far: 1.45x was close at val_mae 7.525804, 1.5x 7.530105, 1.55x 7.538147, 1.75x 7.656927, 1.4x 7.531981, 1.43x 7.526523, 1.47x 7.526719.
- A small HistGradientBoosting blend with the bias-shifted GP improved substantially: 90% GP / 10% HistGradientBoosting reached val_mae 6.987598.
- Increasing the HistGradientBoosting weight kept helping up to 50%: 20% HGB val_mae 6.498735, 30% HGB 6.123640, 50% HGB 5.809324.

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
- Shrinking the bias correction to 0.75x was worse than 1.0x: val_mae 8.115945.
- Changing the calibration holdout from top 20% to top 15% or top 25% was worse: val_mae 7.578010 and 7.530244 respectively.
- Averaging bias estimates across top 15/20/25% holdouts was worse than the single top-20 holdout: val_mae 7.532951.
- TabPFN could not run without noninteractive license token/model access.
- Extrapolation mixup MLP with tilted underprediction loss was much worse: val_mae 17.395464.
- Non-GP guidance attempts so far are below the GP/HGB best: quantile HGB val_mae 14.784369, anchor-point HGB 15.864447, NAM 8.677804.
- Monotonic-family tuning is active per human guidance. Pure positive-weight monotonic MLP was very poor (21.067557); partial monotonic with free path improved but still poor (17.404421); soft gradient-penalty monotonic MLP is best in family so far (11.782069 with penalty 0.2). Reducing penalty to 0.05 worsened to 12.623925.
- Monotonic stagnation: 10 monotonic-family experiments all failed to approach the current best. Stronger penalty 0.5 worsened to 14.259600; feature masks at abs-corr >=0.4 and >=0.15 worsened to 12.826622 and 13.303174; longer training worsened to 12.306579; LR 3e-3 and 7.5e-4 worsened to 13.789898 and 14.383346.

## Structural findings
- The validation split appears sensitive to GP tail behavior; explicit additive linear kernel terms have not improved extrapolation despite guidance suggesting linear+local GP kernels as a strong general OOD baseline.
- The target power transform is beneficial or at least not removable for this current GP.
- The baseline GP underpredicts high-y OOD regions. A constant bias correction learned from top-y train holdout is robust; slope/intercept correction is too aggressive.
- HistGradientBoosting alone underpredicts high-y holdout more than GP on train-only diagnostics, but a small blend improves the final biased GP, likely adding local shape while GP+bias handles extrapolation.
- Hard monotonic architectures appear too restrictive for this dataset; soft monotonic regularization preserves more capacity and is the current monotonic direction to tune.

## Unexplored directions
- Switch away from monotonic after stagnation. Next best non-GP direction is NAM-style models (best non-GP so far val_mae 8.677804) and tune for a sustained run before switching again.
- Try small ensembles/blends of Ridge, GP, and tree/boosting models if available in scikit-learn.
- Try target quantile or shifted/asymmetric residual adjustments aimed at high-y OOD underprediction.
