"""
Autoresearch training script.
Risk Extrapolation (REx) MLP on train-only proxy environments.
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


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RExMLPRegressor:
    def __init__(self, random_state=42, rex_lambda=0.01, n_envs=4):
        self.random_state = random_state
        self.rex_lambda = rex_lambda
        self.n_envs = n_envs
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

    def _make_envs(self, y):
        ranks = pd.qcut(
            pd.Series(y).rank(method="first"),
            q=self.n_envs,
            labels=False,
            duplicates="drop",
        )
        return np.asarray(ranks, dtype=np.int64)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_np = self.x_scaler_.fit_transform(X).astype(np.float32)
        y_np = self.y_scaler_.fit_transform(np.asarray(y).reshape(-1, 1)).ravel().astype(
            np.float32
        )
        env_np = self._make_envs(np.asarray(y))

        x = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        env_t = torch.from_numpy(env_np)
        self.model_ = MLP(x.shape[1])
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_size = 192
        deadline = time.time() + min(TIME_BUDGET - 5, 60)
        step = 0

        while time.time() < deadline and step < 25000:
            idx = torch.randint(0, len(x), (batch_size,))
            xb = x[idx]
            yb = y_t[idx]
            eb = env_t[idx]
            pred = self.model_(xb)

            env_losses = []
            for env_id in range(self.n_envs):
                mask = eb == env_id
                if torch.any(mask):
                    env_losses.append(F.smooth_l1_loss(pred[mask], yb[mask], beta=0.5))

            losses = torch.stack(env_losses)
            mean_loss = losses.mean()
            warmup = min(1.0, step / 2500)
            loss = mean_loss + warmup * self.rex_lambda * losses.var(unbiased=False)

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
print("Model: RExMLPRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = RExMLPRegressor()

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
