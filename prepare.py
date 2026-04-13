"""
One-time data preparation for autoresearch experiments.
Downloads data.

Usage:
    python prepare.py                  # full prep (download + tokenizer)

Data is stored in ~/.cache/autoresearch/.
"""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
MAX_EPOCHS = 100
LABEL_COLUMN = "Concrete compressive strength"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data():
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    dataset = fetch_ucirepo(id=165)
    print(f"Data: downloaded {dataset.name} dataset")
    print("Metadata: ", dataset.metadata)
    print("Variables: ", dataset.variables)

    X = dataset.data.features
    y = dataset.data.targets
    data = pd.concat([X, y], axis=1)
    data = data.sort_values(by=LABEL_COLUMN).reset_index(drop=True)

    train_data = data[:800]
    test_data = data[800:]
    train_data.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test_data.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"Data: split into train and test datasets, shape: {train_data.shape}, {test_data.shape}")
    print(f"Data: saved to {os.path.join(DATA_DIR, 'train.csv')} and {os.path.join(DATA_DIR, 'test.csv')}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model):
    """Evaluate the model on test data."""
    data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X = data.drop(columns=[LABEL_COLUMN])
    y = data[LABEL_COLUMN]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return mae, r2, rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data()
    print()

    print("Done! Ready to train.")