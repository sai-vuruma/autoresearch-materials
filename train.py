"""
Autoresearch training script.
Gated extrapolating MLP on the Concrete dataset.
Usage: uv run train.py
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from prepare import TIME_BUDGET, DATA_DIR, LABEL_COLUMN, evaluate_model


class GatedNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x, gate):
        mlp_pred = self.mlp(x).squeeze(-1)
        linear_pred = self.linear(x).squeeze(-1)
        return (1.0 - gate) * mlp_pred + gate * linear_pred


class GatedExtrapMLPRegressor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_np = self.x_scaler_.fit_transform(X).astype(np.float32)
        y_np = self.y_scaler_.fit_transform(np.asarray(y).reshape(-1, 1)).ravel().astype(
            np.float32
        )
        self.mu_ = x_np.mean(axis=0)
        cov = np.cov(x_np.T) + 1e-4 * np.eye(x_np.shape[1])
        self.cov_inv_ = np.linalg.pinv(cov)
        train_mahal = np.sqrt(
            np.einsum("ij,jk,ik->i", x_np - self.mu_, self.cov_inv_, x_np - self.mu_)
        )
        self.gate_center_ = float(np.quantile(train_mahal, 0.75))
        self.gate_temp_ = 0.75

        x = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        self.model_ = GatedNet(x.shape[1])
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_size = 128
        deadline = time.time() + min(TIME_BUDGET - 5, 60)
        step = 0

        while time.time() < deadline and step < 25000:
            idx = torch.randint(0, len(x), (batch_size,))
            xb = x[idx]
            yb = y_t[idx]
            mlp_pred = self.model_(xb, torch.zeros(batch_size))
            linear_pred = self.model_(xb, torch.ones(batch_size))
            random_gate = torch.rand(batch_size)
            mixed_pred = self.model_(xb, random_gate)
            loss = F.smooth_l1_loss(mlp_pred, yb, beta=0.5)
            loss = loss + 0.5 * F.smooth_l1_loss(linear_pred, yb, beta=0.5)
            loss = loss + 0.5 * F.smooth_l1_loss(mixed_pred, yb, beta=0.5)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 2.0)
            optimizer.step()
            step += 1

        self.num_steps_ = step
        self.model_.eval()
        return self

    def _gate(self, x_np):
        diffs = x_np - self.mu_
        mahal = np.sqrt(np.einsum("ij,jk,ik->i", diffs, self.cov_inv_, diffs))
        return 1.0 / (1.0 + np.exp(-(mahal - self.gate_center_) / self.gate_temp_))

    def predict(self, X):
        x_np = self.x_scaler_.transform(X).astype(np.float32)
        gate_np = self._gate(x_np).astype(np.float32)
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(x_np), torch.from_numpy(gate_np)).numpy()
        return self.y_scaler_.inverse_transform(pred.reshape(-1, 1)).ravel()


t_start = time.time()

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

print("Device: cpu")
print("Model: GatedExtrapMLPRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = GatedExtrapMLPRegressor()

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
