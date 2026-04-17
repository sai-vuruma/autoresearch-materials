## Current best
val_mae: 9.662166 (iter 8, commit 30f172b)
Key config: StandardScaler on inputs and target, exact GaussianProcessRegressor, kernel = Constant * Matern(nu=1.5) + WhiteKernel, `alpha=1e-4`, white noise init `0.3`, 2 optimizer restarts.

## What works
- Replacing the neural baseline with an exact GP is a large win on this small tabular dataset: val_mae improved from 32.208550 to 9.921234 (baseline `fa257c9` -> iter 1 `6f4af2a`).
- Simpler non-neural structure is appropriate here because the current environment is CPU-only and the previous MLP stopped after 100 epochs while barely using the 300s budget (`fa257c9`).
- Slightly stronger GP regularization preserved accuracy while reducing fit time: `alpha=1e-4`, higher white-noise init, and 2 restarts matched/slightly beat the earlier best (`677956a`).
- A rougher Matern prior helped materially: changing from `nu=2.5` to `nu=1.5` improved val_mae from 9.921232 to 9.662166 on the regularized GP (`677956a` -> `30f172b`).

## What doesn't work
- The original MLP baseline is badly underfit in this environment and is not competitive for this dataset (`fa257c9`).
- Adding a linear kernel term plus ARD length scales made the GP both slower and worse: val_mae 14.293646 with a 31.7s fit (`90cd75e`).
- A `log1p` target transform degraded the simple GP baseline to val_mae 12.923762 (`73c40f3`).
- A robust `HistGradientBoostingRegressor` baseline was substantially worse than the GP at val_mae 16.985010 (`82f33ff`).
- Replacing the Matern prior with RationalQuadratic regressed to val_mae 10.562639 (`ec14b8a`).

## Structural findings
- For 800 training rows and 8 numeric features, exact kernel regression is a better starting point than a moderately sized deep MLP.
- OOD-oriented guidance can be followed in a lightweight way here through robust kernel methods before adding more complex neural invariance machinery.
- The current best model is simple for a reason: extra kernel flexibility and tree boosting reduced generalization, while modest regularization plus a rougher single Matern kernel improved it.

## Unexplored directions
- Try other simple Matern smoothness settings or one-step neighborhoods around the new best (`nu=1.5`, `alpha=1e-4`, white noise init `0.3`).
- Try larger jitter/noise floor only if it keeps the `nu=1.5` gain; the same idea worked in moderation but warnings still remain.
- Try a hard-example reweighting/JTT-style second pass only if simple kernel tuning plateaus.
