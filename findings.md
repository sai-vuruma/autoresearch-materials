## Current best
val_mae: 4.990229 (commit 2af03bc, calibration-heavy NAM)
Key config: NAM with per-feature hidden 64 and high-y residual bias multiplier 2.8. This is the lowest logged val_mae but it is calibration-heavy / split-specific. Best less-calibrated GP-HGB circleback is val_mae 5.777955 (commit 618fa60). Clean NAM hidden 64 was val_mae 8.677804.

## What works
- Baseline Matern GP with target power transform is the best so far: val_mae 9.627624.
- Bias-only calibration from a train-only high-y holdout substantially improves OOD validation: val_mae 7.778742. Internal diagnostic showed baseline GP underpredicted top-y training holdout by about 5.26 on average; applying that mean shift generalized better than affine calibration.
- Amplifying the bias correction to 1.25x improved further: val_mae 7.576891.
- Multiplier tuning found 1.445x best so far: 1.45x was close at val_mae 7.525804, 1.5x 7.530105, 1.55x 7.538147, 1.75x 7.656927, 1.4x 7.531981, 1.43x 7.526523, 1.47x 7.526719.
- A small HistGradientBoosting blend with the bias-shifted GP improved substantially: 90% GP / 10% HistGradientBoosting reached val_mae 6.987598.
- Increasing the HistGradientBoosting weight kept helping up to 50%: 20% HGB val_mae 6.498735, 30% HGB 6.123640, 50% HGB 5.809324.
- Circle-back on GP-HGB blend found 55% HGB best in that local sweep at val_mae 5.777955; 60% was 5.784802, 52.5% was 5.792492, and 70% was 5.918737. Still below current NAM best.
- NAM residual-bias sweep reached the lowest logged val_mae 4.990229 at multiplier 2.8, but this should be treated as calibration-heavy rather than a clean architecture win. Clean hidden-96 NAM without residual bias was worse than clean hidden-64 NAM: 8.995952 vs 8.677804.

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
- NALU stagnation after 10 experiments: baseline hidden-32 NALU val_mae 12.687710; hidden 64 12.673314; log-input skip improved to 10.218379; hidden 32/128 with skip regressed; raw-input skip slightly improved to 10.092560; longer training, LR 5e-4/2e-3, and MSE loss regressed; tilted loss was best NALU at 8.393496 but still far below NAM/GP-HGB.
- EDL stagnation after 10 experiments: baseline evidential MLP val_mae 13.973953; lambda 0.001/0.1 worsened; extra gamma MSE worsened; tilted gamma loss helped slightly; smaller hidden sizes helped with hidden 32 best among standard EDL variants at 13.876219; hidden 16 worsened; LR 2e-3 worsened; hidden 32 with LR 5e-4 was best EDL at 12.283950 but still far below current best.
- SNGP/Lipschitz stagnation after 10 experiments: baseline spectral encoder + 256 RFF val_mae 10.697754; direct spectral head collapsed to 20.631965; 512 RFF improved to 9.722637 and was best SNGP; 1024 RFF, latent 128, hidden 64, LR 5e-4/2e-3, and tilted loss all regressed.
- DKL stagnation after 10 experiments: baseline exact gpytorch DKL val_mae 13.334652; lower LR 0.003 barely improved to 13.301977; latent 8/32 regressed; removing latent normalization improved to 9.861672 and was best DKL; 1500 training steps destabilized badly; smaller feature extractor regressed.
- Gated ExtrapMLP stagnation after 10 experiments: baseline gated MLP val_mae 23.816717 because the linear inference path was untrained; training both paths improved to 14.838245; delaying fallback improved slightly to 14.493049 and was best in family. Stronger linear-path loss, 99th-percentile gate centers, smoother gates, empirical-gate training, RidgeCV fallback, earlier RidgeCV fallback, and gated residual-over-Ridge all regressed.

## Structural findings
- The validation split appears sensitive to GP tail behavior; explicit additive linear kernel terms have not improved extrapolation despite guidance suggesting linear+local GP kernels as a strong general OOD baseline.
- The target power transform is beneficial or at least not removable for this current GP.
- The baseline GP underpredicts high-y OOD regions. A constant bias correction learned from top-y train holdout is robust; slope/intercept correction is too aggressive.
- HistGradientBoosting alone underpredicts high-y holdout more than GP on train-only diagnostics, but a small blend improves the final biased GP, likely adding local shape while GP+bias handles extrapolation.
- Hard monotonic architectures appear too restrictive for this dataset; soft monotonic regularization preserves more capacity and is the current monotonic direction to tune.
- NALU arithmetic units are not a good fit here. The only useful ingredient was a simple linear skip path; the arithmetic stack itself did not close the gap.
- EDL uncertainty parameterization is not helping point MAE on this benchmark; it behaves like an underperforming MLP even after mean-loss and size/LR tuning.
- SNGP distance-aware regularization helps less than expected for point MAE; random Fourier head is necessary, but capacity/LR tuning did not close the gap.
- Important correction: the very low NAM results around val_mae 4.99 and 4.67 include high-y residual bias calibration, so they are calibration-heavy and should not be treated as clean structural wins.
- DKL only became reasonable after removing latent normalization, which supports the guidance note that extrapolating means/kernels matter. Still far below simpler GP-HGB and calibrated NAM lines.
- Gated extrapolating MLP is not competitive here. The fallback mechanism mostly trades one underfit neural path for another; using RidgeCV as the fallback did not help, so this family is stagnated.

## Unexplored directions
- Per human guidance: for each new approach, run at least 10 experiments or until crash/stagnation, then circle back to bests. Avoid high-y holdout calibration hacks unless explicitly testing calibration.
- Try small ensembles/blends of Ridge, GP, and tree/boosting models if available in scikit-learn.
- Try target quantile or shifted/asymmetric residual adjustments aimed at high-y OOD underprediction.
