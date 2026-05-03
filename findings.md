## Current best
val_mae: 8.502591 (iter 5, commit e1fb7b0)
Key config: weighted-conformal-style GP, StandardScaler X, Yeo-Johnson y transform, Constant*Matern(nu=1.5)+White kernel, 20% high-y calibration tail, prediction shifted by 1.25x weighted 10/90 signed-residual midpoint.

## What works
- Weighted residual midpoint correction helps OOD high-y extrapolation: baseline blended correction reached 9.377819; using only the 10/90 midpoint improved to 8.915559; scaling midpoint by 1.25 improved to 8.502591.
- Keeping the conformal/calibration correction matters: raw GP prediction regressed badly to 11.691338.

## What doesn't work
- Adding DotProduct plus Matern/RBF local kernels inside the current shifted-GP wrapper regressed to 9.597352.
- Weighted median residual shift alone regressed to 9.942002, so the median residual term is less useful than the 10/90 midpoint.

## Structural findings
- The current OOD split appears dominated by upward target shift; post-hoc signed residual correction is currently higher leverage than changing the GP kernel.
- The best correction so far is a scaled tail-midpoint signed residual, not the original median-plus-midpoint blend.

## Unexplored directions
- Continue tuning midpoint scale around 1.25, including 1.5 and intermediate values.
- Tune the signed residual quantile pair, e.g. 0.05/0.95 or 0.2/0.9.
- Try delta-learning or linear residual priors if shift scaling plateaus.
