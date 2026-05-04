## Current best
val_mae: 4.861591 (commit 4138a03, clean NAM)
Key config: NAM with per-feature hidden 64, tau 0.995 tilted underprediction loss, and tilted-loss coefficient 18.0. This uses no high-y residual-bias correction and is now the best logged result. The prior residual-biased NAM line reached 4.990229 at commit 2af03bc and the widened residual-biased NAM reached 4.671150 at commit b569090, but both are calibration-heavy / split-specific and should not be treated as clean architecture wins. Best GP-HGB bias-shift circleback is val_mae 5.777955 (commit 618fa60), also calibration-heavy.

## What works
- Baseline Matern GP with target power transform established the starting point: val_mae 9.627624.
- Bias-only calibration from a train-only high-y holdout substantially improves OOD validation: val_mae 7.778742. Internal diagnostic showed baseline GP underpredicted top-y training holdout by about 5.26 on average; applying that mean shift generalized better than affine calibration.
- Amplifying the bias correction to 1.25x improved further: val_mae 7.576891.
- Multiplier tuning found 1.445x best so far: 1.45x was close at val_mae 7.525804, 1.5x 7.530105, 1.55x 7.538147, 1.75x 7.656927, 1.4x 7.531981, 1.43x 7.526523, 1.47x 7.526719.
- A small HistGradientBoosting blend with the bias-shifted GP improved substantially: 90% GP / 10% HistGradientBoosting reached val_mae 6.987598.
- Increasing the HistGradientBoosting weight kept helping up to 50%: 20% HGB val_mae 6.498735, 30% HGB 6.123640, 50% HGB 5.809324.
- Circle-back on GP-HGB blend found 55% HGB best in that local sweep at val_mae 5.777955; 60% was 5.784802, 52.5% was 5.792492, and 70% was 5.918737. Still below current NAM best.
- NAM residual-bias sweep reached val_mae 4.990229 at multiplier 2.8, and residual-biased width tuning reached 4.671150, but those should be treated as calibration-heavy rather than clean architecture wins. Clean hidden-96 NAM without residual bias was worse than clean hidden-64 NAM before the asymmetric-loss circle-back: 8.995952 vs 8.677804.
- Clean NAM circle-back is the strongest non-gaming result: hidden 64 with tau 0.995 and tilted-loss coefficient 18.0 reached val_mae 4.861591, val_r2 0.468339, val_rmse 6.366535. This beat the GP-HGB bias-shift line and the earlier residual-biased NAM 2af03bc without using a high-y residual correction.

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
- Early non-GP guidance attempts were below the GP/HGB best: quantile HGB val_mae 14.784369, anchor-point HGB 15.864447, initial clean NAM 8.677804. Later clean NAM circle-back reversed this with val_mae 4.861591.
- Monotonic-family tuning is active per human guidance. Pure positive-weight monotonic MLP was very poor (21.067557); partial monotonic with free path improved but still poor (17.404421); soft gradient-penalty monotonic MLP is best in family so far (11.782069 with penalty 0.2). Reducing penalty to 0.05 worsened to 12.623925.
- Monotonic stagnation: 10 monotonic-family experiments all failed to approach the current best. Stronger penalty 0.5 worsened to 14.259600; feature masks at abs-corr >=0.4 and >=0.15 worsened to 12.826622 and 13.303174; longer training worsened to 12.306579; LR 3e-3 and 7.5e-4 worsened to 13.789898 and 14.383346.
- NALU stagnation after 10 experiments: baseline hidden-32 NALU val_mae 12.687710; hidden 64 12.673314; log-input skip improved to 10.218379; hidden 32/128 with skip regressed; raw-input skip slightly improved to 10.092560; longer training, LR 5e-4/2e-3, and MSE loss regressed; tilted loss was best NALU at 8.393496 but still far below NAM/GP-HGB.
- EDL stagnation after 10 experiments: baseline evidential MLP val_mae 13.973953; lambda 0.001/0.1 worsened; extra gamma MSE worsened; tilted gamma loss helped slightly; smaller hidden sizes helped with hidden 32 best among standard EDL variants at 13.876219; hidden 16 worsened; LR 2e-3 worsened; hidden 32 with LR 5e-4 was best EDL at 12.283950 but still far below current best.
- SNGP/Lipschitz stagnation after 10 experiments: baseline spectral encoder + 256 RFF val_mae 10.697754; direct spectral head collapsed to 20.631965; 512 RFF improved to 9.722637 and was best SNGP; 1024 RFF, latent 128, hidden 64, LR 5e-4/2e-3, and tilted loss all regressed.
- DKL stagnation after 10 experiments: baseline exact gpytorch DKL val_mae 13.334652; lower LR 0.003 barely improved to 13.301977; latent 8/32 regressed; removing latent normalization improved to 9.861672 and was best DKL; 1500 training steps destabilized badly; smaller feature extractor regressed.
- Gated ExtrapMLP stagnation after 10 experiments: baseline gated MLP val_mae 23.816717 because the linear inference path was untrained; training both paths improved to 14.838245; delaying fallback improved slightly to 14.493049 and was best in family. Stronger linear-path loss, 99th-percentile gate centers, smoother gates, empirical-gate training, RidgeCV fallback, earlier RidgeCV fallback, and gated residual-over-Ridge all regressed.
- REx/IRM-style tuning after 15 experiments: four target-quantile proxy environments were poor, with lambda 1.0/0.1/0.01 yielding 16.745895/16.143206/16.076845. Two proxy environments were better; lambda 1.0 reached 13.980288. Reducing capacity helped: hidden 64 reached 13.613668, hidden 32 reached 12.265790, and hidden 16 with lambda 3.0 was best at 11.200681. Hidden 8 collapsed and lambda 5.0 regressed, so the family is plateaued far below GP-HGB.
- Delta Ridge MLP stagnation after 10 experiments: baseline RidgeCV plus residual MLP reached 12.625052. Hidden 32 and 128 regressed; SmoothL1 beta 0.5/0.1 and MSE regressed; residual scale 0.5/1.5 regressed; weight decay 1e-3 regressed. Lowering LR to 5e-4 was best at 11.844453, still worse than compact REx and far below GP-HGB.
- TTT/auxiliary masked-reconstruction tuning after 21 experiments: source training with SSL helped, but actual test-time adaptation hurt. Baseline ssl_weight 0.5/mask 0.30/adapt 50 was 12.491432; no adaptation improved to 10.919581; light adaptation still worsened. Mask 0.20 with ssl_weight 0.4 improved to 9.800508. Adding a latent bottleneck was the main gain: hidden 64/latent 16/no adaptation reached 8.891929; weight decay 1e-3 gave the best TTT result at 8.878820. Latent 8, hidden 32, adaptation, lower LR, and weight decay 3e-3 regressed.
- JTT/Group-DRO fallback stagnation after 10 experiments: baseline hard_frac 0.2/hard_weight 5.0 was 15.625766. Reducing hard weight helped up to weight 1.5 at 13.245479, but weight 1.2 regressed. Hard fractions 0.1 and 0.3 regressed. Short warmup was the only useful lever: warmup 4000 reached 13.019778 and warmup 2000 was best at 11.370241; warmup 1000 and hard_weight 2.0 with warmup 2000 regressed.
- Clean NAM circle-back after residual-bias correction: increasing tilted-loss tau from 0.68 to 0.995 improved clean hidden-64 NAM from 8.677804 to 8.272579 without any residual bias term. Keeping tau 0.995 and increasing the tilted-loss coefficient was the main clean win: weights 0.25/0.35/0.50/0.75/1.0/1.5/3.0/5.0/7.5/10.0/15.0/20.0 reached 7.959747/7.787637/7.557193/7.153369/6.898865/6.604386/6.282530/5.761117/5.622784/5.436261/4.963022/4.926122. Weight 30.0 regressed to 5.020619, 22.5 regressed to 4.963906, 16.5 regressed to 4.950137, and 19.0 regressed to 5.160120. The best clean NAM is weight 18.0 at commit 4138a03 with val_mae 4.861591, val_r2 0.468339, val_rmse 6.366535.
- Clean NAM post-bracket checks: hidden 80 and 48 regressed to 5.129987 and 5.056971; LR 1e-3 and 3e-3 regressed to 4.956465 and 5.081329; weight decay 1e-4 regressed to 4.934147. Restored train.py to the best clean NAM configuration from commit 4138a03.

## Structural findings
- The validation split appears sensitive to GP tail behavior; explicit additive linear kernel terms have not improved extrapolation despite guidance suggesting linear+local GP kernels as a strong general OOD baseline.
- The target power transform is beneficial or at least not removable for this current GP.
- The baseline GP underpredicts high-y OOD regions. A constant bias correction learned from top-y train holdout is robust; slope/intercept correction is too aggressive.
- HistGradientBoosting alone underpredicts high-y holdout more than GP on train-only diagnostics, but a small blend improves the final biased GP, likely adding local shape while GP+bias handles extrapolation.
- Hard monotonic architectures appear too restrictive for this dataset; soft monotonic regularization preserves more capacity and is the current monotonic direction to tune.
- NALU arithmetic units are not a good fit here. The only useful ingredient was a simple linear skip path; the arithmetic stack itself did not close the gap.
- EDL uncertainty parameterization is not helping point MAE on this benchmark; it behaves like an underperforming MLP even after mean-loss and size/LR tuning.
- SNGP distance-aware regularization helps less than expected for point MAE; random Fourier head is necessary, but capacity/LR tuning did not close the gap.
- Important correction: the very low earlier NAM results around val_mae 4.99 and 4.67 include high-y residual bias calibration, so those specific commits are calibration-heavy and should not be treated as clean structural wins. The current clean NAM best at 4.861591 does not use that residual-bias mechanism.
- DKL only became reasonable after removing latent normalization, which supports the guidance note that extrapolating means/kernels matter. Still far below simpler GP-HGB and calibrated NAM lines.
- Gated extrapolating MLP is not competitive here. The fallback mechanism mostly trades one underfit neural path for another; using RidgeCV as the fallback did not help, so this family is stagnated.
- REx with train-only target proxy environments acts mostly like regularization. Compact networks help, but the approach still underpredicts the high-y OOD tail and does not approach the clean NAM circle-back.
- Delta Ridge MLP did not realize the guidance ranking on this split. The residual MLP appears to damage OOD extrapolation more often than it helps; plain Ridge extrapolation plus learned residuals stays in the 11.8+ MAE range.
- TTT's useful ingredient here is auxiliary masked reconstruction as representation regularization, not test-time updates. The best TTT run is a clean structural neural result, but still below the clean NAM circle-back.
- JTT hard-sample reweighting is mostly harmful; the hard samples selected by early ERM loss do not behave like a useful OOD proxy group for this target-sorted split.
- Clean NAM with a very strong asymmetric training loss now beats the earlier GP-HGB bias-shift line and the prior residual-biased NAM commit 2af03bc. This is a training-objective change, not a high-y residual correction, so it directly addresses the residual-gaming concern.

## Unexplored directions
- Per human guidance: for each new approach, run at least 10 experiments or until crash/stagnation, then circle back to bests. Avoid high-y holdout calibration hacks unless explicitly testing calibration.
- Try small ensembles/blends of Ridge, GP, and tree/boosting models if available in scikit-learn.
- If continuing NAM, fine-tune the clean asymmetric objective near tau 0.995 / tilt weight 18.0, or try ensembles that preserve the no-residual-bias constraint.
