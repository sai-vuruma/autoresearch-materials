## Current best
val_mae: 7.802070 (iter 9, commit bf92903)
Key config: weighted-conformal-style GP, StandardScaler X, Yeo-Johnson y transform, Constant*Matern(nu=1.5)+White kernel, 20% high-y calibration tail, prediction shifted by 2.2x weighted 10/90 signed-residual midpoint.

## What works
- Weighted residual midpoint correction helps OOD high-y extrapolation: baseline blended correction reached 9.377819; using only the 10/90 midpoint improved to 8.915559; scaling midpoint by 1.25, 1.5, 2.0, and 2.2 improved to 8.502591, 8.182734, 7.837565, and 7.802070.
- Keeping the conformal/calibration correction matters: raw GP prediction regressed badly to 11.691338.

## What doesn't work
- Adding DotProduct plus Matern/RBF local kernels inside the current shifted-GP wrapper regressed to 9.597352.
- Weighted median residual shift alone regressed to 9.942002, so the median residual term is less useful than the 10/90 midpoint.
- Scaling the midpoint to 2.5 overshot and regressed to 7.935612; 2.3 also regressed slightly to 7.826833 versus 2.2.

## Structural findings
- The current OOD split appears dominated by upward target shift; post-hoc signed residual correction is currently higher leverage than changing the GP kernel.
- The best correction so far is a scaled tail-midpoint signed residual, not the original median-plus-midpoint blend.

## Unexplored directions
- Continue fine tuning midpoint scale around 2.1 to 2.25.
- Tune the signed residual quantile pair, e.g. 0.05/0.95 or 0.2/0.9.
- Try delta-learning or linear residual priors if shift scaling plateaus.
