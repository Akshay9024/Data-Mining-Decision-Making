from __future__ import annotations

import json
import logging
import os
import random
import urllib.request
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
_STATS = ["mean", "se", "worst"]
# WDBC column order: 10 means, then 10 standard errors, then 10 "worst" values.
FEATURE_NAMES = [f"{f}_{s}" for s in _STATS for f in _BASE_FEATURES]
COLUMN_NAMES = ["id", "diagnosis"] + FEATURE_NAMES


def load_config(path: str = "configs/config.json") -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def download_if_missing(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Downloading dataset: %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)


def load_raw(config: dict[str, Any]) -> pd.DataFrame:
    dest = config["paths"]["data_file"]
    download_if_missing(config["data"]["download_url"], dest)
    df = pd.read_csv(dest, header=None, names=COLUMN_NAMES)
    n_features = config["data"]["n_features"]
    assert df.shape == (569, n_features + 2), f"unexpected shape {df.shape}"
    assert df.isna().sum().sum() == 0, "missing values present"
    assert set(df["diagnosis"].unique()) == {"M", "B"}, "unexpected diagnosis labels"
    return df


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop the non-predictive id column and encode M=1 (positive), B=0."""
    X = df.drop(columns=["id", "diagnosis"])
    y = (df["diagnosis"] == "M").astype(int).to_numpy()
    return X, y


def load_split(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Deterministic stratified train/test split, identical across scripts given the seed."""
    X, y = prepare(load_raw(config))
    return train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        stratify=y,
        random_state=config["seed"],
    )


def run_eda(config: dict[str, Any]) -> None:
    fig_dir = config["paths"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    df = load_raw(config)
    X, y = prepare(df)

    counts = df["diagnosis"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(counts.index, counts.values, color=["#4c72b0", "#c44e52"])
    ax.set_title("Class distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=150)
    plt.close(fig)

    corr = X.corr()
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=6)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation (Pearson)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(6, 5, figsize=(18, 18))
    for ax, feat in zip(axes.ravel(), FEATURE_NAMES):
        ax.hist(X.loc[y == 0, feat], bins=30, alpha=0.6, label="B", color="#4c72b0")
        ax.hist(X.loc[y == 1, feat], bins=30, alpha=0.6, label="M", color="#c44e52")
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=6)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Per-feature distributions by class")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_distributions.png"), dpi=120)
    plt.close(fig)
    logger.info("EDA artifacts written to %s", fig_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    set_seed(config["seed"])
    X_train, X_test, y_train, y_test = load_split(config)
    logger.info(
        "train=%d test=%d train_pos_rate=%.4f test_pos_rate=%.4f",
        len(y_train), len(y_test), y_train.mean(), y_test.mean(),
    )
    run_eda(config)


if __name__ == "__main__":
    main()
