"""
Autoresearch training script.
Delta Ridge MLP on the Concrete dataset.
Usage: uv run train.py
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from prepare import TIME_BUDGET, DATA_DIR, LABEL_COLUMN, evaluate_model


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeltaRidgeMLPRegressor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.residual_scale = 0.5
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_np = self.x_scaler_.fit_transform(X).astype(np.float32)
        y_np = self.y_scaler_.fit_transform(np.asarray(y).reshape(-1, 1)).ravel().astype(
            np.float32
        )
        self.ridge_ = RidgeCV(alphas=np.logspace(-4, 4, 17)).fit(x_np, y_np)
        base_np = self.ridge_.predict(x_np).astype(np.float32)
        residual_np = y_np - base_np

        x = torch.from_numpy(x_np)
        residual_t = torch.from_numpy(residual_np)
        self.model_ = ResidualMLP(x.shape[1])
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_size = 128
        deadline = time.time() + min(TIME_BUDGET - 5, 60)
        step = 0

        while time.time() < deadline and step < 25000:
            idx = torch.randint(0, len(x), (batch_size,))
            pred = self.model_(x[idx])
            loss = F.smooth_l1_loss(pred, residual_t[idx], beta=0.25)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 2.0)
            optimizer.step()
            step += 1

        self.num_steps_ = step
        self.model_.eval()
        return self

    def predict(self, X):
        x_np = self.x_scaler_.transform(X).astype(np.float32)
        base = self.ridge_.predict(x_np)
        with torch.no_grad():
            residual = self.model_(torch.from_numpy(x_np)).numpy()
        pred = base + self.residual_scale * residual
        return self.y_scaler_.inverse_transform(pred.reshape(-1, 1)).ravel()


t_start = time.time()

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

print("Device: cpu")
print("Model: DeltaRidgeMLPRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = DeltaRidgeMLPRegressor()

t_start_training = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - t_start_training

mae, r2, rmse = evaluate_model(model)
t_end = time.time()

num_params = sum(p.numel() for p in model.model_.parameters())

print("---")
print(f"val_mae:          {mae:.6f}")
print(f"val_r2:           {r2:.6f}")
print(f"val_rmse:         {rmse:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print("peak_vram_mb:     0.0")
print(f"num_steps:        {model.num_steps_}")
print(f"num_params:       {num_params:,}")
print("num_epochs:       0")
