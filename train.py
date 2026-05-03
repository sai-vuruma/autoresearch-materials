"""
Autoresearch training script.
Just Train Twice (JTT) hard-sample reweighted MLP.
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


class JTTMLPRegressor:
    def __init__(self, random_state=42, hard_frac=0.2, hard_weight=1.5):
        self.random_state = random_state
        self.hard_frac = hard_frac
        self.hard_weight = hard_weight
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

    def _train_model(self, x, y_t, weights, steps, lr=1e-3):
        model = MLP(x.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        batch_size = 128
        step = 0
        while step < steps:
            idx = torch.randint(0, len(x), (batch_size,))
            pred = model(x[idx])
            per_sample = F.smooth_l1_loss(pred, y_t[idx], beta=0.5, reduction="none")
            loss = (per_sample * weights[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            step += 1
        model.eval()
        return model

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_np = self.x_scaler_.fit_transform(X).astype(np.float32)
        y_np = self.y_scaler_.fit_transform(np.asarray(y).reshape(-1, 1)).ravel().astype(
            np.float32
        )
        x = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        base_weights = torch.ones(len(x))

        deadline = time.time() + min(TIME_BUDGET - 5, 60)
        warmup_steps = 8000
        train_steps = 17000
        self.warm_model_ = self._train_model(x, y_t, base_weights, warmup_steps)

        with torch.no_grad():
            losses = F.smooth_l1_loss(self.warm_model_(x), y_t, beta=0.5, reduction="none")
        threshold = torch.quantile(losses, 1.0 - self.hard_frac)
        weights = torch.ones(len(x))
        weights[losses >= threshold] = self.hard_weight
        weights = weights / weights.mean()

        remaining = max(1000, min(train_steps, int((deadline - time.time()) * 1000)))
        self.model_ = self._train_model(x, y_t, weights, remaining)
        self.num_steps_ = warmup_steps + remaining
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
print("Model: JTTMLPRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = JTTMLPRegressor()

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
