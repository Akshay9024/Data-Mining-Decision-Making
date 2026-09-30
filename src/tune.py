from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.data import load_config, load_split, set_seed

logger = logging.getLogger(__name__)

STATIC_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist"}
TUNED_KEYS = [
    "max_depth", "min_child_weight", "gamma", "learning_rate",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "scale_pos_weight",
]


def build_estimator(
    params: dict[str, Any],
    n_estimators: int,
    config: dict[str, Any],
    early_stopping_rounds: int | None = None,
) -> XGBClassifier:
    return XGBClassifier(
        **STATIC_PARAMS,
        **params,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        device=config["device"],
        n_jobs=config["n_jobs"],
        random_state=config["seed"],
        verbosity=0,
    )


def cv_evaluate(
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    n_estimators: int,
    early_stopping_rounds: int | None,
) -> dict[str, float]:
    """Stratified k-fold CV on the training partition only. Returns mean/std metrics."""
    # ponytail: early stopping picks best_iteration on the same val fold it is scored on,
    # so the CV metric is mildly optimistic; test set (touched once) is the honest number.
    # Upgrade to nested CV only if the CV<->test gap proves to matter.
    skf = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=config["seed"])
    aucs: list[float] = []
    accs: list[float] = []
    iters: list[int] = []
    for tr, va in skf.split(X, y):
        model = build_estimator(params, n_estimators, config, early_stopping_rounds)
        if early_stopping_rounds:
            model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            best = int(model.best_iteration)
            proba = model.predict_proba(X[va], iteration_range=(0, best + 1))[:, 1]
            iters.append(best + 1)
        else:
            model.fit(X[tr], y[tr], verbose=False)
            proba = model.predict_proba(X[va])[:, 1]
            iters.append(n_estimators)
        aucs.append(roc_auc_score(y[va], proba))
        accs.append(accuracy_score(y[va], (proba >= 0.5).astype(int)))
    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "mean_best_iter": int(round(float(np.mean(iters)))),
    }


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _sample_stage_a(rng: np.random.Generator, space: dict[str, Any], spw_choices: list[float]) -> dict[str, Any]:
    return {
        "max_depth": int(rng.integers(space["max_depth"][0], space["max_depth"][1] + 1)),
        "min_child_weight": int(rng.integers(space["min_child_weight"][0], space["min_child_weight"][1] + 1)),
        "gamma": float(rng.uniform(*space["gamma"])),
        "learning_rate": _loguniform(rng, *space["learning_rate"]),
        "subsample": float(rng.uniform(*space["subsample"])),
        "colsample_bytree": float(rng.uniform(*space["colsample_bytree"])),
        "reg_alpha": _loguniform(rng, *space["reg_alpha"]),
        "reg_lambda": _loguniform(rng, *space["reg_lambda"]),
        "scale_pos_weight": float(rng.choice(spw_choices)),
    }


def run_stage_a(
    X: np.ndarray, y: np.ndarray, config: dict[str, Any], spw_choices: list[float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = config["tuning"]["stage_a"]
    rng = np.random.default_rng(config["seed"])
    max_n = config["tuning"]["max_n_estimators"]
    esr = config["tuning"]["early_stopping_rounds"]
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for i in range(cfg["n_iter"]):
        params = _sample_stage_a(rng, cfg["space"], spw_choices)
        res = cv_evaluate(params, X, y, config, max_n, esr)
        trials.append({"params": params, **res})
        if best is None or res["roc_auc_mean"] > best["roc_auc_mean"]:
            best = {"params": params, **res}
        logger.info("stage-A %d/%d auc=%.4f", i + 1, cfg["n_iter"], res["roc_auc_mean"])
    assert best is not None
    return best, trials


def run_stage_b(
    X: np.ndarray, y: np.ndarray, config: dict[str, Any], seed_params: dict[str, Any], spw_choices: list[float]
) -> optuna.Study:
    """TPE refinement over the Stage-A space, seeded (enqueued) with the Stage-A best."""
    space = config["tuning"]["stage_a"]["space"]
    max_n = config["tuning"]["max_n_estimators"]
    esr = config["tuning"]["early_stopping_rounds"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1]),
            "min_child_weight": trial.suggest_int("min_child_weight", space["min_child_weight"][0], space["min_child_weight"][1]),
            "gamma": trial.suggest_float("gamma", space["gamma"][0], space["gamma"][1]),
            "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True),
            "subsample": trial.suggest_float("subsample", space["subsample"][0], space["subsample"][1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", space["colsample_bytree"][0], space["colsample_bytree"][1]),
            "reg_alpha": trial.suggest_float("reg_alpha", space["reg_alpha"][0], space["reg_alpha"][1], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", space["reg_lambda"][0], space["reg_lambda"][1], log=True),
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", spw_choices),
        }
        res = cv_evaluate(params, X, y, config, max_n, esr)
        trial.set_user_attr("mean_best_iter", res["mean_best_iter"])
        trial.set_user_attr("accuracy_mean", res["accuracy_mean"])
        return res["roc_auc_mean"]

    db = config["paths"]["study_db"]
    if os.path.exists(db):
        os.remove(db)  # fresh, reproducible study each full run; persisted afterwards
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=config["paths"]["study_name"],
        storage=f"sqlite:///{db}",
        load_if_exists=False,
    )
    study.enqueue_trial({k: seed_params[k] for k in TUNED_KEYS})
    study.optimize(objective, n_trials=config["tuning"]["stage_b"]["n_trials"])
    return study


def _plot_curve(grid: list[float], mean: list[float], std: list[float], axis: str, path: str, logx: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(grid, mean, yerr=std, marker="o", capsize=3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(axis)
    ax.set_ylabel("CV ROC-AUC (mean ± std)")
    ax.set_title(f"Sensitivity: {axis}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sensitivity_curves(
    X: np.ndarray, y: np.ndarray, config: dict[str, Any], best_params: dict[str, Any]
) -> None:
    grids = config["sensitivity"]
    fig_dir = config["paths"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    max_n = config["tuning"]["max_n_estimators"]
    esr = config["tuning"]["early_stopping_rounds"]
    out: dict[str, Any] = {}
    for axis in ("max_depth", "learning_rate", "n_estimators"):
        means: list[float] = []
        stds: list[float] = []
        for v in grids[axis]:
            params = dict(best_params)
            if axis == "n_estimators":
                res = cv_evaluate(params, X, y, config, int(v), None)  # fixed rounds, no early stopping
            else:
                params[axis] = v
                res = cv_evaluate(params, X, y, config, max_n, esr)
            means.append(res["roc_auc_mean"])
            stds.append(res["roc_auc_std"])
        out[axis] = {"grid": grids[axis], "mean": means, "std": stds}
        _plot_curve(grids[axis], means, stds, axis, os.path.join(fig_dir, f"sensitivity_{axis}.png"),
                    logx=axis in ("learning_rate", "n_estimators"))
        logger.info("sensitivity %s done", axis)
    with open(os.path.join(config["paths"]["results_dir"], "sensitivity.json"), "w") as fh:
        json.dump(out, fh, indent=2)


def run_tuning(config: dict[str, Any]) -> None:
    set_seed(config["seed"])
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    X_train, _, y_train, _ = load_split(config)  # test partition never used here
    X = X_train.to_numpy()
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    spw_choices = [1.0, round(spw, 4)]
    logger.info("scale_pos_weight choices: %s", spw_choices)

    best_a, trials = run_stage_a(X, y_train, config, spw_choices)
    logger.info("stage-A best auc=%.4f params=%s", best_a["roc_auc_mean"], best_a["params"])

    study = run_stage_b(X, y_train, config, best_a["params"], spw_choices)
    best_params = {k: study.best_params[k] for k in TUNED_KEYS}
    best_n_estimators = int(study.best_trial.user_attrs["mean_best_iter"])
    logger.info("stage-B best auc=%.4f n_estimators=%d params=%s",
                study.best_value, best_n_estimators, best_params)

    sensitivity_curves(X, y_train, config, best_params)

    record = {
        "params": best_params,
        "n_estimators": best_n_estimators,
        "cv_roc_auc": float(study.best_value),
        "cv_accuracy": float(study.best_trial.user_attrs["accuracy_mean"]),
        "scale_pos_weight_positive": spw,
        "seed": config["seed"],
    }
    with open(config["paths"]["best_params_file"], "w") as fh:
        json.dump(record, fh, indent=2)
    with open(os.path.join(config["paths"]["results_dir"], "stage_a_trials.json"), "w") as fh:
        json.dump(trials, fh, indent=2)
    logger.info("best config written to %s", config["paths"]["best_params_file"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    args = ap.parse_args()
    run_tuning(load_config(args.config))


if __name__ == "__main__":
    main()
