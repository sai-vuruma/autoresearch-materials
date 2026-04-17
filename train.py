"""
Autoresearch training script.
Small-data tabular regression baseline using an exact Gaussian Process.
Usage: uv run train.py
"""

import os
import time

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from prepare import TIME_BUDGET, DATA_DIR, LABEL_COLUMN, evaluate_model


t_start = time.time()

# Load training data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

print("Device: cpu")
print("Model: GaussianProcessRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

# A Matern kernel is a strong default for small OOD-prone tabular regression.
kernel = (
    ConstantKernel(1.0, (1e-2, 1e3))
    * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
    + WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-3, 1e2))
)

model = TransformedTargetRegressor(
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
                    random_state=42,
                ),
            ),
        ]
    ),
    transformer=StandardScaler(),
)

t_start_training = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - t_start_training

mae, r2, rmse = evaluate_model(model)
t_end = time.time()

fitted_gp = model.regressor_.named_steps["gp"]
num_hyperparams = fitted_gp.kernel_.theta.size

print("---")
print(f"val_mae:          {mae:.6f}")
print(f"val_r2:           {r2:.6f}")
print(f"val_rmse:         {rmse:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print("peak_vram_mb:     0.0")
print("num_steps:        1")
print(f"num_params:       {num_hyperparams}")
print("num_epochs:       1")
