## Current best
val_mae: 6.296263 (iter 22, commit c78bfca)
Key config: weighted-conformal-style GP, StandardScaler X, Yeo-Johnson y transform, Constant*Matern(nu=1.5)+White kernel, 31% high-y calibration tail (248 samples), prediction shifted by 2.15x weighted 10/90 signed-residual midpoint.

## What works
- Weighted residual midpoint correction helps OOD high-y extrapolation: baseline blended correction reached 9.377819; using only the 10/90 midpoint improved to 8.915559; scaling midpoint by 1.25, 1.5, 2.0, 2.2, and 2.15 improved to 8.502591, 8.182734, 7.837565, 7.802070, and 7.798539.
- Keeping the conformal/calibration correction matters: raw GP prediction regressed badly to 11.691338.
- Increasing the calibration tail from 20% to 30% is a large win with the 2.15x midpoint shift: val_mae improved from 7.798539 to 6.588122 and val_r2 became positive.
- Fine tuning calibration to 31% improved further to 6.296263; this is the best current setting.

## What doesn't work
- Adding DotProduct plus Matern/RBF local kernels inside the current shifted-GP wrapper regressed to 9.597352.
- Weighted median residual shift alone regressed to 9.942002, so the median residual term is less useful than the 10/90 midpoint.
- Scaling the midpoint to 2.5 overshot and regressed to 7.935612; 2.3 also regressed slightly to 7.826833 versus 2.2.
- Changing residual midpoint quantiles away from 0.10/0.90 hurt badly: 0.05/0.95 regressed to 9.316308 and 0.20/0.80 regressed to 9.162711.
- Calibration tail size is sensitive: 25% regressed badly to 10.547205, 35% to 6.787869, 40% to 6.731566, and 120 samples to 8.775288.
- Nearby calibration sizes around 31% did not beat it: 28% got 6.864895, 32% got 6.411387, 244 samples got 6.396944, and 252 samples got 6.394749.

## Structural findings
- The current OOD split appears dominated by upward target shift; post-hoc signed residual correction is currently higher leverage than changing the GP kernel.
- The best correction so far is a scaled tail-midpoint signed residual, not the original median-plus-midpoint blend.
- The 31% calibration tail appears to balance GP fit data and tail residual estimation better than nearby tested fractions.

## Unexplored directions
- Continue fine tuning midpoint scale only if needed; 2.15 is the best tested local value, with 2.1 and 2.17 slightly worse.
- Try asymmetric quantile pairs only if they preserve one endpoint at 0.10 or 0.90; symmetric widening/narrowing failed.
- Fine tune exact calibration sample count around 248 if continuing parameter search, e.g. 246, 247, 249, 250.
- Try delta-learning or linear residual priors if calibration tuning plateaus.
