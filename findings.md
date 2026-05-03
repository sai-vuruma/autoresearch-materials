## Current best
val_mae: 9.377819 (commit ba121d3)
Key config: split the sorted training set into a GP fit block and a top-tail calibration block, fit the incumbent Yeo-Johnson target-transform GP on the fit block, then apply weighted conformal-style signed-residual correction using a domain classifier trained on calibration vs test covariates.

## What works
- Replacing the neural baseline with an exact GP is a large win on this small tabular dataset: val_mae improved from 32.208550 to 9.921234 (baseline `1988f07` -> iter 1 `32f95e4`).
- Simpler non-neural structure is appropriate here because the current environment is CPU-only and the previous MLP stopped after 100 epochs while barely using the 300s budget (`1988f07`).
- Slightly stronger GP regularization preserved accuracy while reducing fit time: `alpha=1e-4`, higher white-noise init, and 2 restarts matched/slightly beat the earlier best (`f502724`).
- A rougher Matern prior helped materially: changing from `nu=2.5` to `nu=1.5` improved val_mae from 9.921232 to 9.662166 on the regularized GP (`f502724` -> `61fcde9`).
- A Yeo-Johnson transform on the target improved the GP again after a long structural plateau: val_mae moved from 9.662166 to 9.627625 (`61fcde9` -> `e4a39af`).
- A weighted conformal-style shift correction on top of the best GP produced the first clear post-GP breakthrough: val_mae improved from 9.627624 to 9.377819 by using a held-out calibration tail and domain-classifier weights against the unlabeled test covariates (`e6d1a84` -> `ba121d3`).

## What doesn't work
- The original MLP baseline is badly underfit in this environment and is not competitive for this dataset (`1988f07`).
- Replacing the Matern prior with RationalQuadratic regressed to val_mae 10.562639 (`1e4ab26`).
- Making the kernel even rougher (`nu=0.5`) collapsed performance to val_mae 12.364519 (`95a668c`).
- Increasing explicit GP jitter from `1e-4` to `1e-3` produced no measurable gain over the current best (`5e957d3`).
- JTT-style residual reweighting with a second GP fit regressed to val_mae 10.243857 (`f7a0cb4`).
- Increasing GP restart count from 2 to 8 only made the fit slower without improving val_mae (`2640e19`).
- Lowering white-noise initialization from `0.3` to `0.1` also tied the incumbent (`5d23fde`).
- A simple GP-plus-ridge ensemble regressed badly to val_mae 12.876797 (`6599c50`).
- Training a supervised MLP encoder and fitting the GP on its learned latent features regressed badly to val_mae 14.918993 (`f236159`).
- An age-binned REx MLP, a GP with age-bin indicator features, ExtraTrees, k-NN, kernel ridge, SVR, and shallow Huber boosting all regressed materially (`f3dc612`, `94fd26a`, `1c6e542`, `96d7e47`, `3ff80b9`, `79b60da`, `1190213`).
- A Yeo-Johnson transform on the inputs regressed badly even though the same transform on the target helped (`895c49b`).
- Within target-side transforms, `log1p` and quantile-normalized Gaussian targets were bad regressions, while Box-Cox was close but still worse than Yeo-Johnson (`993f4ea`, `eda7bc5`, `d988118`).
- Explicit OOD neural objectives still underperform badly even with pseudo-environments from the shifted features: Group DRO, REx, IRM, TTT, and SNGP all regressed materially (`d31f38d`, `4cad859`, `acd8399`, `64631e2`, `b496115`).
- Hybrid GP combinations without shift-aware calibration did not improve the primary metric: blends, gates, and residual stacks all tied or regressed (`8645254`, `3f597a7`, `e23a952`, `db1658d`, `9c85fe0`, `33d625b`, `87f44fb`, `2e68667`, `77e9d65`).

## Structural findings
- For 800 training rows and 8 numeric features, exact kernel regression is a better starting point than a moderately sized deep MLP.
- OOD-oriented guidance can be followed in a lightweight way here through robust kernel methods before adding more complex neural invariance machinery.
- The current best model is simple for a reason: extra kernel flexibility hurt generalization, while modest regularization plus a rougher single Matern kernel improved it.
- The local optimum seems fairly narrow: `nu=1.5` helps, but both smoother (`2.5`) and much rougher (`0.5`) priors are worse.
- The current input-space GP is in a local plateau: recent changes to restarts, weighting, normalization, and shallow ensembling all failed to improve `61fcde9`.
- A learned latent space did not help here: replacing raw standardized features with a small supervised MLP encoder made the downstream GP much worse, so lack of learned representation is not the obvious bottleneck.
- Stagnation note: there are now 10 consecutive discarded experiments since `61fcde9`. The exhausted space includes GP hyperparameter neighborhood tuning, simple reweighting, shallow ensembling, learned-latent GP, REx-style MLP, and environment-indicator feature augmentation. The bottleneck looks structural rather than a missed local GP setting.
- The broader non-GP search also looks poor so far: tree ensembles, neighbor methods, and other kernel machines have all been substantially worse than the best GP.
- Target geometry matters more than input geometry here: transforming `y` helped the GP while transforming `X` hurt it.
- Among target transforms, Yeo-Johnson appears to be the robust choice: more aggressive alternatives either regressed sharply or only matched part of the improvement profile.
- The split is true extrapolation in target space: test strengths start above the training maximum. Pure representation-learning OOD methods were weak here, but post-hoc covariate-shift correction on top of the GP did help.
- The useful signal from `guidance.md` was not the neural architecture suggestions but the weighted-calibration idea: using unlabeled test covariates to reweight calibration residuals corrected the incumbent GP in the right direction.

## Unexplored directions
- Refine the weighted-calibration recipe around `ba121d3`: calibration split location, weighting model, and how the signed residual correction is turned into a point shift.
- Combine the weighted correction with a simpler full-train GP refit if the split-fit penalty from holding out calibration data turns out to be avoidable.
- TabPFN-style in-context regression remains untested only because `tabpfn` is not installed and adding dependencies is disallowed by `program.md`.
