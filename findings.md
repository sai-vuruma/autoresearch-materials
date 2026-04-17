## Current best
val_mae: 9.921234 (iter 1, commit 32f95e4)
Key config: StandardScaler on inputs and target, exact GaussianProcessRegressor, kernel = Constant * Matern(nu=2.5) + WhiteKernel, 5 optimizer restarts.

## What works
- Replacing the neural baseline with an exact GP is a large win on this small tabular dataset: val_mae improved from 32.208550 to 9.921234 (baseline `1988f07` -> iter 1 `32f95e4`).
- Simpler non-neural structure is appropriate here because the current environment is CPU-only and the previous MLP stopped after 100 epochs while barely using the 300s budget (`1988f07`).

## What doesn't work
- The original MLP baseline is badly underfit in this environment and is not competitive for this dataset (`1988f07`).

## Structural findings
- For 800 training rows and 8 numeric features, exact kernel regression is a better starting point than a moderately sized deep MLP.
- OOD-oriented guidance can be followed in a lightweight way here through robust kernel methods before adding more complex neural invariance machinery.

## Unexplored directions
- Tune the GP kernel family: RationalQuadratic, additive linear + Matern, ARD length scales, larger jitter/noise floor.
- Try robust target transforms such as log1p if label distribution is skewed.
- Try a hard-example reweighting/JTT-style second pass only if kernel tuning plateaus.
