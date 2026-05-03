"""
Autoresearch training script.
GP regression on the Concrete Compressive Strength dataset.
Usage: uv run train.py
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from prepare import TIME_BUDGET, DATA_DIR, LABEL_COLUMN, evaluate_model


class GPRegressor:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def _make_model(self):
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

    def fit(self, X, y):
        self.model_ = self._make_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class BiasShiftedGPRegressor(GPRegressor):
    def fit(self, X, y):
        y_array = np.asarray(y)
        order = np.argsort(y_array)
        n_train = int(len(y_array) * 0.8)
        train_idx = order[:n_train]
        calibration_idx = order[n_train:]

        calibration_model = self._make_model()
        calibration_model.fit(X.iloc[train_idx], y_array[train_idx])
        calibration_pred = calibration_model.predict(X.iloc[calibration_idx])
        self.bias_ = 1.445 * float(np.mean(calibration_pred - y_array[calibration_idx]))

        self.model_ = self._make_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X) - self.bias_


t_start = time.time()

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

print("Device: cpu")
print("Model: BiasShiftedGPRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = BiasShiftedGPRegressor()

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
print("num_steps:        1")
print("num_params:       4")
print("num_epochs:       1")
