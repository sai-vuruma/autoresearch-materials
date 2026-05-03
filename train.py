"""
Autoresearch training script.
GP regression on the Concrete Compressive Strength dataset.
Usage: uv run train.py
"""

import os
import time

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


t_start = time.time()

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

print("Device: cpu")
print("Model: GPRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = GPRegressor()

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
