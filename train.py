"""
Autoresearch training script. Single-GPU (or CPU), single-file.
Simple MLP for regression on the Concrete Compressive Strength dataset.
Usage: uv run train.py
"""

import os
import gc
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from prepare import TIME_BUDGET, MAX_EPOCHS, DATA_DIR, LABEL_COLUMN, evaluate_model

# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------

@dataclass
class MLPConfig:
    input_dim: int = 8
    hidden_dims: tuple = (256, 256, 128)
    dropout: float = 0.1


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dims = [config.input_dim, *config.hidden_dims, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.GELU())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class ModelWrapper:
    """Wraps a trained PyTorch MLP with a sklearn-compatible predict() interface."""

    def __init__(self, model, scaler, y_scaler, device):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.device = device

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().reshape(-1, 1)
        preds = self.y_scaler.inverse_transform(preds).ravel()
        return preds


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_DIMS = (256, 256, 128)   # hidden layer sizes
DROPOUT = 0.1                   # dropout probability

# Optimization
BATCH_SIZE = 64                 # mini-batch size
LR = 1e-3                       # peak learning rate (AdamW)
WEIGHT_DECAY = 1e-4             # L2 regularisation
GRAD_CLIP = 1.0                 # gradient norm clip
WARMUP_RATIO = 0.05             # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.4            # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.01            # final LR as fraction of peak

# ---------------------------------------------------------------------------
# Setup: data, model, optimizer
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load training data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X_train = train_df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)
y_train = train_df[LABEL_COLUMN].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

dataset = TensorDataset(
    torch.tensor(X_scaled, dtype=torch.float32),
    torch.tensor(y_scaled, dtype=torch.float32),
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

config = MLPConfig(input_dim=X_train.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
print(f"Model config: {asdict(config)}")

model = MLP(config).to(device)
num_params = model.num_params()
print(f"Parameters: {num_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(f"Time budget:      {TIME_BUDGET}s")
print(f"Training samples: {len(X_train):,}")

# ---------------------------------------------------------------------------
# LR schedule (warmup → flat → warmdown)
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return max(cooldown, FINAL_LR_FRAC)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
smooth_train_loss = 0.0
step = 0
epoch = 0

while True:
    for x_batch, y_batch in loader:
        t0 = time.time()

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = F.mse_loss(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # Update LR
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for pg in optimizer.param_groups:
            pg["lr"] = LR * lrm

        train_loss_f = loss.item()

        # Fast fail: abort if loss explodes or goes NaN
        if math.isnan(train_loss_f) or train_loss_f > 1e8:
            print("FAIL")
            exit(1)

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:06d} ({pct_done:.1f}%) | "
            f"loss(MSE): {debiased:.4f} | "
            f"lrm: {lrm:.3f} | "
            f"dt: {dt * 1000:.1f}ms | "
            f"epoch: {epoch} | "
            f"remaining: {remaining:.0f}s    ",
            end="", flush=True,
        )

        # GC management (freeze after first step to avoid GC stalls)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

        step += 1

        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    epoch += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

    if epoch >= MAX_EPOCHS:
        break
print()  # newline after \r training log

# ---------------------------------------------------------------------------
# Final eval
# ---------------------------------------------------------------------------

wrapper = ModelWrapper(model, scaler, y_scaler, device)
mae, r2, rmse = evaluate_model(wrapper)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0

print("---")
print(f"val_mae:          {mae:.6f}")
print(f"val_r2:           {r2:.6f}")
print(f"val_rmse:         {rmse:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params:,}")
print(f"num_epochs:       {epoch}")
