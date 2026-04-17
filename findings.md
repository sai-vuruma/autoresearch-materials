## Current best
val_mae: 9.921234 (iter 1, commit 6f4af2a)
Key config: StandardScaler on inputs and target, exact GaussianProcessRegressor, kernel = Constant * Matern(nu=2.5) + WhiteKernel, 5 optimizer restarts.

## What works
- Replacing the neural baseline with an exact GP is a large win on this small tabular dataset: val_mae improved from 32.208550 to 9.921234 (baseline `fa257c9` -> iter 1 `6f4af2a`).
- Simpler non-neural structure is appropriate here because the current environment is CPU-only and the previous MLP stopped after 100 epochs while barely using the 300s budget (`fa257c9`).

## What doesn't work
- The original MLP baseline is badly underfit in this environment and is not competitive for this dataset (`fa257c9`).
- Adding a linear kernel term plus ARD length scales made the GP both slower and worse: val_mae 14.293646 with a 31.7s fit (`90cd75e`).
- A `log1p` target transform degraded the simple GP baseline to val_mae 12.923762 (`73c40f3`).
- A robust `HistGradientBoostingRegressor` baseline was substantially worse than the GP at val_mae 16.985010 (`82f33ff`).

## Structural findings
- For 800 training rows and 8 numeric features, exact kernel regression is a better starting point than a moderately sized deep MLP.
- OOD-oriented guidance can be followed in a lightweight way here through robust kernel methods before adding more complex neural invariance machinery.
- The current best model is simple for a reason: extra kernel flexibility and tree boosting both reduced generalization, so the next gains likely need better regularized kernel structure rather than more capacity.

## Unexplored directions
- Tune the GP kernel family: RationalQuadratic, additive linear + Matern, ARD length scales, larger jitter/noise floor.
- Try larger jitter/noise floor or fewer optimizer restarts to address the recurrent GP numerical warnings without changing the inductive bias.
- Try RationalQuadratic or a different smoothness choice while keeping the single-kernel structure simple.
- Try a hard-example reweighting/JTT-style second pass only if simple kernel tuning plateaus.
