## Current best
val_mae: 9.662166 (iter 4, commit 61fcde9)
Key config: StandardScaler on inputs and target, exact GaussianProcessRegressor, kernel = Constant * Matern(nu=1.5) + WhiteKernel, `alpha=1e-4`, white noise init `0.3`, 2 optimizer restarts.

## What works
- Replacing the neural baseline with an exact GP is a large win on this small tabular dataset: val_mae improved from 32.208550 to 9.921234 (baseline `1988f07` -> iter 1 `32f95e4`).
- Simpler non-neural structure is appropriate here because the current environment is CPU-only and the previous MLP stopped after 100 epochs while barely using the 300s budget (`1988f07`).
- Slightly stronger GP regularization preserved accuracy while reducing fit time: `alpha=1e-4`, higher white-noise init, and 2 restarts matched/slightly beat the earlier best (`f502724`).
- A rougher Matern prior helped materially: changing from `nu=2.5` to `nu=1.5` improved val_mae from 9.921232 to 9.662166 on the regularized GP (`f502724` -> `61fcde9`).

## What doesn't work
- The original MLP baseline is badly underfit in this environment and is not competitive for this dataset (`1988f07`).
- Replacing the Matern prior with RationalQuadratic regressed to val_mae 10.562639 (`1e4ab26`).

## Structural findings
- For 800 training rows and 8 numeric features, exact kernel regression is a better starting point than a moderately sized deep MLP.
- OOD-oriented guidance can be followed in a lightweight way here through robust kernel methods before adding more complex neural invariance machinery.
- The current best model is simple for a reason: extra kernel flexibility hurt generalization, while modest regularization plus a rougher single Matern kernel improved it.

## Unexplored directions
- Try other simple Matern smoothness settings around the new best (`nu=1.5`) to see whether the optimum is broad or narrow.
- Try larger jitter/noise floor only if it keeps the same simple kernel and materially improves stability.
- Try robust target transforms such as log1p if label distribution is skewed.
- Try a hard-example reweighting/JTT-style second pass only if kernel tuning plateaus.
