"""
Autoresearch training script.
Test-time-training MLP with masked feature reconstruction.
Usage: uv run train.py
"""

import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from prepare import TIME_BUDGET, DATA_DIR, LABEL_COLUMN, evaluate_model


class TTTNet(nn.Module):
    def __init__(self, in_dim, hidden=64, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent),
            nn.SiLU(),
        )
        self.reg_head = nn.Linear(latent, 1)
        self.ssl_head = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def predict_y(self, x):
        return self.reg_head(self.encode(x)).squeeze(-1)

    def reconstruct_x(self, x):
        return self.ssl_head(self.encode(x))


class TTTMLPRegressor:
    def __init__(self, random_state=42, ssl_weight=0.4, mask_prob=0.2, adapt_steps=0):
        self.random_state = random_state
        self.ssl_weight = ssl_weight
        self.mask_prob = mask_prob
        self.adapt_steps = adapt_steps
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

    def _masked_reconstruction_loss(self, x):
        mask = (torch.rand_like(x) < self.mask_prob).float()
        masked_x = x * (1.0 - mask)
        recon = self.model_.reconstruct_x(masked_x)
        denom = mask.sum().clamp_min(1.0)
        return (((recon - x) * mask).pow(2).sum() / denom)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_np = self.x_scaler_.fit_transform(X).astype(np.float32)
        y_np = self.y_scaler_.fit_transform(np.asarray(y).reshape(-1, 1)).ravel().astype(
            np.float32
        )
        x = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        self.model_ = TTTNet(x.shape[1])
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=1e-3, weight_decay=3e-3)
        batch_size = 128
        deadline = time.time() + min(TIME_BUDGET - 5, 60)
        step = 0

        while time.time() < deadline and step < 25000:
            idx = torch.randint(0, len(x), (batch_size,))
            xb = x[idx]
            pred = self.model_.predict_y(xb)
            reg_loss = F.smooth_l1_loss(pred, y_t[idx], beta=0.5)
            ssl_loss = self._masked_reconstruction_loss(xb)
            loss = reg_loss + self.ssl_weight * ssl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 2.0)
            optimizer.step()
            step += 1

        self.num_steps_ = step
        self.model_.eval()
        return self

    def _adapt(self, x):
        encoder_state = copy.deepcopy(self.model_.encoder.state_dict())
        ssl_state = copy.deepcopy(self.model_.ssl_head.state_dict())
        self.model_.encoder.train()
        self.model_.ssl_head.train()
        self.model_.reg_head.eval()
        optimizer = torch.optim.AdamW(
            list(self.model_.encoder.parameters()) + list(self.model_.ssl_head.parameters()),
            lr=1e-4,
            weight_decay=0.0,
        )
        for _ in range(self.adapt_steps):
            loss = self._masked_reconstruction_loss(x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model_.encoder.parameters()) + list(self.model_.ssl_head.parameters()),
                2.0,
            )
            optimizer.step()
        self.model_.eval()
        return encoder_state, ssl_state

    def predict(self, X):
        x_np = self.x_scaler_.transform(X).astype(np.float32)
        x = torch.from_numpy(x_np)
        encoder_state, ssl_state = self._adapt(x)
        with torch.no_grad():
            pred = self.model_.predict_y(x).numpy()
        self.model_.encoder.load_state_dict(encoder_state)
        self.model_.ssl_head.load_state_dict(ssl_state)
        self.model_.eval()
        return self.y_scaler_.inverse_transform(pred.reshape(-1, 1)).ravel()


t_start = time.time()

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

print("Device: cpu")
print("Model: TTTMLPRegressor")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

model = TTTMLPRegressor()

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
