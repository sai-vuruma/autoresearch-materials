"""
Autoresearch training script.
Neural Additive Model on the Concrete dataset.
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


class NAM(nn.Module):
    def __init__(self, in_dim, hidden=80):
        super().__init__()
        self.feature_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(in_dim)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        terms = [net(x[:, i : i + 1]).squeeze(-1) for i, net in enumerate(self.feature_nets)]
        return torch.stack(terms, dim=0).sum(dim=0) + self.bias


class NAMRegressor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

    def _tilted_loss(self, pred, target, tau=0.995):
        err = target - pred
        return torch.maximum(tau * err, (tau - 1.0) * err).mean()

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_np = self.x_scaler_.fit_transform(X).astype(np.float32)
        y_np = self.y_scaler_.fit_transform(np.asarray(y).reshape(-1, 1)).ravel().astype(
            np.float32
        )
        x = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)

        self.model_ = NAM(x.shape[1])
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=2e-3, weight_decay=2e-4)
        batch_size = 128
        deadline = time.time() + min(TIME_BUDGET - 5, 60)
        step = 0

        while time.time() < deadline and step < 25000:
            idx = torch.randint(0, len(x), (batch_size,))
            pred = self.model_(x[idx])
            huber = F.smooth_l1_loss(pred, y_t[idx], beta=0.5)
            loss = huber + 18.0 * self._tilted_loss(pred, y_t[idx])

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
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(x_np)).numpy()
        return self.y_scaler_.inverse_transform(pred.reshape(-1, 1)).ravel()


t_start = time.time()

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

print("Device: cpu")
print("Model: NAMRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = NAMRegressor()

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
