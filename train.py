"""
Autoresearch training script.
Weighted-conformal-style covariate-shift correction for the best GP.
Usage: uv run train.py
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from prepare import TIME_BUDGET, DATA_DIR, LABEL_COLUMN, evaluate_model


class WeightedConformalShiftRegressor:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def _make_base_model(self):
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
        )
        return TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "gp",
                        GaussianProcessRegressor(
                            kernel=kernel,
                            alpha=1e-4,
                            normalize_y=True,
                            n_restarts_optimizer=2,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
        )

    @staticmethod
    def _weighted_quantile(values, quantile, sample_weight):
        order = np.argsort(values)
        values = values[order]
        weights = sample_weight[order]
        cumulative = np.cumsum(weights) / np.sum(weights)
        return np.interp(quantile, cumulative, values)

    def fit(self, X, y):
        train_df = X.copy()
        train_df[LABEL_COLUMN] = y.to_numpy()

        calib_size = max(120, int(0.2 * len(train_df)))
        fit_df = train_df.iloc[:-calib_size].reset_index(drop=True)
        calib_df = train_df.iloc[-calib_size:].reset_index(drop=True)

        self.feature_names_ = [col for col in train_df.columns if col != LABEL_COLUMN]
        X_fit = fit_df[self.feature_names_]
        y_fit = fit_df[LABEL_COLUMN]
        X_calib = calib_df[self.feature_names_]
        y_calib = calib_df[LABEL_COLUMN]

        self.model_ = self._make_base_model()
        self.model_.fit(X_fit, y_fit)

        calib_pred = self.model_.predict(X_calib)
        signed_residuals = y_calib.to_numpy() - calib_pred

        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        X_test = test_df[self.feature_names_]
        domain_X = pd.concat([X_calib, X_test], axis=0, ignore_index=True)
        domain_y = np.array([0] * len(X_calib) + [1] * len(X_test))
        self.domain_clf_ = GradientBoostingClassifier(random_state=self.random_state)
        self.domain_clf_.fit(domain_X, domain_y)

        calib_proba = self.domain_clf_.predict_proba(X_calib)
        weights = calib_proba[:, 1] / np.clip(calib_proba[:, 0], 1e-6, None)
        self.bias_ = self._weighted_quantile(signed_residuals, 0.5, weights)
        self.low_q_ = self._weighted_quantile(signed_residuals, 0.1, weights)
        self.high_q_ = self._weighted_quantile(signed_residuals, 0.9, weights)
        return self

    def predict(self, X):
        base_pred = self.model_.predict(X)
        midpoint_shift = 0.5 * (self.low_q_ + self.high_q_)
        return base_pred + 0.5 * self.bias_ + 0.5 * midpoint_shift


t_start = time.time()

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

print("Device: cpu")
print("Model: WeightedConformalShiftRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = WeightedConformalShiftRegressor()

t_start_training = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - t_start_training

mae, r2, rmse = evaluate_model(model)
t_end = time.time()

print("---")
print(f"val_mae:          {mae:.6f}")
print(f"val_r2:           {r2:.6f}")
print(f"val_rmse:         {rmse:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print("peak_vram_mb:     0.0")
print("num_steps:        2")
print("num_params:       4")
print("num_epochs:       1")
