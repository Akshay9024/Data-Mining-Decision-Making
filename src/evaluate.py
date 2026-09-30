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


def train_oof_proba(
    X: np.ndarray, y: np.ndarray, params: dict[str, Any], n_estimators: int, config: dict[str, Any]
) -> np.ndarray:
    """Out-of-fold predicted probabilities on the train partition. All thresholds derive from these."""
    skf = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=config["seed"])
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        model = build_estimator(params, n_estimators, config, None)
        model.fit(X[tr], y[tr], verbose=False)
        oof[va] = model.predict_proba(X[va])[:, 1]
    return oof


def youden_threshold(oof: np.ndarray, y: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, oof)
    return float(thr[int(np.argmax(tpr - fpr))])


def _class_rates(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    fn_rate = fn / (fn + tp) if (fn + tp) else 0.0   # miss rate
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0   # fall-out
    return fn_rate, fp_rate


def cost_threshold(oof: np.ndarray, y: np.ndarray, cost_ratio: float, grid: np.ndarray) -> float:
    """Minimize c_FN*FN_rate + c_FP*FP_rate with c_FP=1, c_FN=cost_ratio (per-class rates)."""
    best_t, best_c = float(grid[0]), np.inf
    for t in grid:
        fn_rate, fp_rate = _class_rates(y, (oof >= t).astype(int))
        cost = cost_ratio * fn_rate + fp_rate
        if cost < best_c:  # ascending grid -> ties resolve to the more recall-favorable (lower) t
            best_c, best_t = cost, float(t)
    return best_t


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


def operating_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    yhat = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, yhat)),
        "recall_M": float(recall_score(y, yhat, zero_division=0)),
        "precision_M": float(precision_score(y, yhat, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_M": float(f1_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _pctile(a: np.ndarray) -> list[float]:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def bootstrap_ci(
    y: np.ndarray, proba: np.ndarray, thresholds: dict[str, float], n: int, seed: int
) -> dict[str, Any]:
    """Percentile bootstrap 95% CIs. Threshold-free (AUCs) + per-operating-point metrics."""
    rng = np.random.default_rng(seed)
    N = len(y)
    auc: list[float] = []
    prauc: list[float] = []
    per = {name: {"accuracy": [], "recall_M": [], "precision_M": [], "f1_M": [], "specificity": []}
           for name in thresholds}
    for _ in range(n):
        idx = rng.integers(0, N, N)
        yb, pb = y[idx], proba[idx]
        if np.unique(yb).size < 2:
            continue
        auc.append(roc_auc_score(yb, pb))
        prauc.append(average_precision_score(yb, pb))
        for name, t in thresholds.items():
            tn, fp, fn, tp = confusion_matrix(yb, (pb >= t).astype(int), labels=[0, 1]).ravel()
            per[name]["accuracy"].append((tp + tn) / yb.size)
            per[name]["recall_M"].append(tp / (tp + fn) if (tp + fn) else 0.0)
            per[name]["precision_M"].append(tp / (tp + fp) if (tp + fp) else 0.0)
            per[name]["f1_M"].append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            per[name]["specificity"].append(tn / (tn + fp) if (tn + fp) else 0.0)
    out: dict[str, Any] = {
        "n_resamples": n,
        "roc_auc": _pctile(np.asarray(auc)),
        "pr_auc": _pctile(np.asarray(prauc)),
        "operating_points": {name: {k: _pctile(np.asarray(v)) for k, v in per[name].items()}
                             for name in thresholds},
    }
    return out


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


def _plot_confusion(cm: dict[str, int], name: str, threshold: float, fig_dir: str) -> None:
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
    ax.set_title(f"Confusion: {name} (t={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"confusion_matrix_{name}.png"), dpi=150)
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


def fn_error_analysis(
    model: XGBClassifier, X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, proba: np.ndarray, threshold: float,
    fig_dir: str, results_dir: str,
) -> list[int]:
    """Per-case analysis of malignant cases missed at `threshold` (native TreeSHAP breakdown)."""
    yhat = (proba >= threshold).astype(int)
    fn_idx = np.where((y_test == 1) & (yhat == 0))[0]
    if fn_idx.size == 0:
        logger.info("no false negatives at threshold %.4f", threshold)
        with open(os.path.join(results_dir, "fn_analysis.json"), "w") as fh:
            json.dump({"threshold": float(threshold), "n_false_negatives": 0, "cases": []}, fh, indent=2)
        return []
    contribs = model.get_booster().predict(xgb.DMatrix(X_test[fn_idx]), pred_contribs=True)
    base = float(contribs[0, -1])
    sv = contribs[:, :-1]
    cases: list[dict[str, Any]] = []
    for j, i in enumerate(fn_idx):
        order = np.argsort(np.abs(sv[j]))[::-1]
        top = order[:12]
        vals = sv[j][top]
        names = [FEATURE_NAMES[k] for k in top]
        colors = ["#c44e52" if v > 0 else "#4c72b0" for v in vals]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.arange(len(top))[::-1], vals, color=colors)
        ax.set_yticks(np.arange(len(top))[::-1])
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("SHAP contribution (log-odds)  |  blue pushes benign, red pushes malignant")
        ax.set_title(f"FN test#{i}  p(M)={proba[i]:.3f}  (base margin {base:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fn_case_{i}_contrib.png"), dpi=150)
        plt.close(fig)

        top5 = order[:5]
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        feat_records: dict[str, Any] = {}
        for ax, k in zip(axes, top5):
            feat = FEATURE_NAMES[k]
            val = float(X_test[i, k])
            ax.hist(X_train[y_train == 0, k], bins=25, alpha=0.6, color="#4c72b0", label="B (train)")
            ax.hist(X_train[y_train == 1, k], bins=25, alpha=0.6, color="#c44e52", label="M (train)")
            ax.axvline(val, color="k", lw=2, label="FN value")
            ax.set_title(feat, fontsize=9)
            ax.tick_params(labelsize=6)
            feat_records[feat] = {
                "value": val,
                "shap": float(sv[j][k]),
                "pctile_in_benign_train": float((X_train[y_train == 0, k] <= val).mean() * 100),
                "pctile_in_malignant_train": float((X_train[y_train == 1, k] <= val).mean() * 100),
            }
        axes[0].legend(fontsize=7)
        fig.suptitle(f"FN test#{i}: top-5 SHAP features vs train class distributions")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fn_case_{i}_features.png"), dpi=120)
        plt.close(fig)

        cases.append({"test_index": int(i), "proba_M": float(proba[i]),
                      "base_margin": base, "top_features": feat_records})
    with open(os.path.join(results_dir, "fn_analysis.json"), "w") as fh:
        json.dump({"threshold": float(threshold), "n_false_negatives": int(fn_idx.size), "cases": cases},
                  fh, indent=2)
    return fn_idx.tolist()


def write_comparison(metrics: dict[str, Any], config: dict[str, Any]) -> None:
    op = metrics["operating_points"]["youden"]
    path = os.path.join(config["paths"]["results_dir"], "comparison_table.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "test_accuracy", "roc_auc", "pr_auc", "recall_M", "precision_M", "f1_M", "mcc"])
        w.writerow(["XGBoost", op["accuracy"], metrics["roc_auc"], metrics["pr_auc"],
                    op["recall_M"], op["precision_M"], op["f1_M"], op["mcc"]])
    logger.info("comparison table with XGBoost row at %s (append baseline rows externally)", path)


def run_evaluation(config: dict[str, Any]) -> None:
    set_seed(config["seed"])
    fig_dir = config["paths"]["figures_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    with open(config["paths"]["best_params_file"]) as fh:
        record = json.load(fh)
    params = record["params"]
    n_estimators = record["n_estimators"]

    X_train_df, X_test_df, y_train, y_test = load_split(config)
    X_train, X_test = X_train_df.to_numpy(), X_test_df.to_numpy()

    # All thresholds are chosen on train out-of-fold probabilities, then frozen.
    oof = train_oof_proba(X_train, y_train, params, n_estimators, config)
    tcfg = config["threshold"]
    grid = np.arange(tcfg["grid_lo"], tcfg["grid_hi"] + tcfg["grid_step"] / 2, tcfg["grid_step"])
    thresholds: dict[str, float] = {"default_0.5": 0.5, "youden": youden_threshold(oof, y_train)}
    for r in tcfg["cost_ratios"]:
        thresholds[f"cost_{r}"] = cost_threshold(oof, y_train, float(r), grid)
    logger.info("operating thresholds: %s", {k: round(v, 4) for k, v in thresholds.items()})

    model = fit_final(X_train, y_train, params, n_estimators, config)
    model.save_model(config["paths"]["model_file"])
    loss_trajectory(X_train, y_train, params, n_estimators, config,
                    os.path.join(fig_dir, "loss_trajectory.png"))

    proba = model.predict_proba(X_test)[:, 1]  # ONE-TOUCH test eval
    operating_points = {name: operating_metrics(y_test, proba, t) for name, t in thresholds.items()}
    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "operating_points": operating_points,
        "bootstrap_ci": bootstrap_ci(y_test, proba, thresholds, config["bootstrap"]["n_test"], config["seed"]),
    }
    with open(config["paths"]["metrics_file"], "w") as fh:
        json.dump(metrics, fh, indent=2)

    _plot_roc(y_test, proba, fig_dir)
    _plot_pr(y_test, proba, fig_dir)
    _plot_calibration(y_test, proba, fig_dir)
    for name, op in operating_points.items():
        _plot_confusion(op["confusion_matrix"], name, thresholds[name], fig_dir)
    top = shap_artifacts(model, X_train, fig_dir)
    fn = fn_error_analysis(model, X_train, y_train, X_test, y_test, proba,
                           thresholds["youden"], fig_dir, results_dir)
    write_comparison(metrics, config)

    logger.info("test roc_auc=%.4f pr_auc=%.4f", metrics["roc_auc"], metrics["pr_auc"])
    logger.info("operating points: %s", {k: {"acc": round(v["accuracy"], 4), "recall": round(v["recall_M"], 4),
                                             "fn": v["confusion_matrix"]["fn"]} for k, v in operating_points.items()})
    logger.info("SHAP top-5: %s | false negatives (youden): %s", top, fn)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    args = ap.parse_args()
    run_evaluation(load_config(args.config))


if __name__ == "__main__":
    main()
