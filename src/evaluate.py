from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from xgboost import XGBClassifier

from src.data import FEATURE_NAMES, load_config, load_split, set_seed
from src.tune import build_estimator

logger = logging.getLogger(__name__)


def youden_threshold(
    X: np.ndarray, y: np.ndarray, params: dict[str, Any], n_estimators: int, config: dict[str, Any]
) -> float:
    """Threshold maximizing Youden's J on out-of-fold train predictions. No test contact."""
    skf = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=config["seed"])
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        model = build_estimator(params, n_estimators, config, None)
        model.fit(X[tr], y[tr], verbose=False)
        oof[va] = model.predict_proba(X[va])[:, 1]
    fpr, tpr, thr = roc_curve(y, oof)
    return float(thr[int(np.argmax(tpr - fpr))])


def fit_final(
    X: np.ndarray, y: np.ndarray, params: dict[str, Any], n_estimators: int, config: dict[str, Any]
) -> XGBClassifier:
    model = build_estimator(params, n_estimators, config, None)
    model.fit(X, y, verbose=False)
    return model


def loss_trajectory(
    X: np.ndarray, y: np.ndarray, params: dict[str, Any], n_estimators: int, config: dict[str, Any], path: str
) -> None:
    """Diagnostic only: train/val logloss vs boosting round on a split of the TRAIN partition."""
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config["seed"])
    model = build_estimator(params, n_estimators, config, None)
    model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    res = model.evals_result()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res["validation_0"]["logloss"], label="train")
    ax.plot(res["validation_1"]["logloss"], label="validation")
    ax.set_xlabel("boosting round")
    ax.set_ylabel("logloss")
    ax.set_title("Training / validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_test(
    model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray, threshold: float
) -> tuple[dict[str, Any], np.ndarray]:
    proba = model.predict_proba(X_test)[:, 1]
    yhat = (proba >= threshold).astype(int)
    yhat05 = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, yhat)),
        "precision_M": float(precision_score(y_test, yhat)),
        "recall_M": float(recall_score(y_test, yhat)),
        "specificity": float(tn / (tn + fp)),
        "f1_M": float(f1_score(y_test, yhat)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "mcc": float(matthews_corrcoef(y_test, yhat)),
        "brier": float(brier_score_loss(y_test, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy_at_0.5": float(accuracy_score(y_test, yhat05)),
        "recall_M_at_0.5": float(recall_score(y_test, yhat05)),
    }
    return metrics, proba


def _plot_roc(y: np.ndarray, proba: np.ndarray, fig_dir: str) -> None:
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, proba):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)


def _plot_pr(y: np.ndarray, proba: np.ndarray, fig_dir: str) -> None:
    prec, rec, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"PR-AUC={average_precision_score(y, proba):.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)


def _plot_confusion(cm: dict[str, int], fig_dir: str) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], ["B (0)", "M (1)"])
    ax.set_yticks([0, 1], ["B (0)", "M (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center")
    ax.set_title("Confusion matrix (Youden threshold)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def _plot_calibration(y: np.ndarray, proba: np.ndarray, fig_dir: str) -> None:
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "calibration.png"), dpi=150)
    plt.close(fig)


def shap_artifacts(model: XGBClassifier, X_train: np.ndarray, fig_dir: str) -> list[str]:
    # Native XGBoost TreeSHAP (margin space); avoids shap 0.46 <-> xgboost 3.x
    # base_score parsing incompatibility. Last column is the bias term, dropped.
    contribs = model.get_booster().predict(xgb.DMatrix(X_train), pred_contribs=True)
    sv = contribs[:, :-1]
    plt.figure()
    shap.summary_plot(sv, X_train, feature_names=FEATURE_NAMES, show=False)
    plt.savefig(os.path.join(fig_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    top_idx = np.argsort(np.abs(sv).mean(axis=0))[::-1][:5]
    for idx in top_idx:
        plt.figure()
        shap.dependence_plot(int(idx), sv, X_train, feature_names=FEATURE_NAMES, show=False)
        plt.savefig(os.path.join(fig_dir, f"shap_dependence_{FEATURE_NAMES[idx]}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    return [FEATURE_NAMES[i] for i in top_idx]


def write_comparison(metrics: dict[str, Any], config: dict[str, Any]) -> None:
    path = os.path.join(config["paths"]["results_dir"], "comparison_table.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "test_accuracy", "roc_auc", "pr_auc", "recall_M", "precision_M", "f1_M", "mcc"])
        w.writerow(["XGBoost", metrics["accuracy"], metrics["roc_auc"], metrics["pr_auc"],
                    metrics["recall_M"], metrics["precision_M"], metrics["f1_M"], metrics["mcc"]])
    logger.info("comparison table with XGBoost row at %s (append baseline rows externally)", path)


def run_evaluation(config: dict[str, Any]) -> None:
    set_seed(config["seed"])
    fig_dir = config["paths"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    with open(config["paths"]["best_params_file"]) as fh:
        record = json.load(fh)
    params = record["params"]
    n_estimators = record["n_estimators"]

    X_train_df, X_test_df, y_train, y_test = load_split(config)
    X_train, X_test = X_train_df.to_numpy(), X_test_df.to_numpy()

    threshold = youden_threshold(X_train, y_train, params, n_estimators, config)
    logger.info("Youden threshold = %.4f", threshold)

    model = fit_final(X_train, y_train, params, n_estimators, config)
    model.save_model(config["paths"]["model_file"])
    loss_trajectory(X_train, y_train, params, n_estimators, config,
                    os.path.join(fig_dir, "loss_trajectory.png"))

    metrics, proba = evaluate_test(model, X_test, y_test, threshold)  # ONE-TOUCH test eval
    with open(config["paths"]["metrics_file"], "w") as fh:
        json.dump(metrics, fh, indent=2)

    _plot_roc(y_test, proba, fig_dir)
    _plot_pr(y_test, proba, fig_dir)
    _plot_confusion(metrics["confusion_matrix"], fig_dir)
    _plot_calibration(y_test, proba, fig_dir)
    top = shap_artifacts(model, X_train, fig_dir)
    write_comparison(metrics, config)

    logger.info("test metrics: %s", json.dumps(metrics))
    logger.info("SHAP top-5 features: %s", top)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    args = ap.parse_args()
    run_evaluation(load_config(args.config))


if __name__ == "__main__":
    main()
