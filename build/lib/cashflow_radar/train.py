"""End-to-end training: data -> features -> baselines -> tuned XGBoost -> calibration ->
threshold -> policy simulation -> SHAP -> artifacts + MLflow.

    python -m cashflow_radar.train                 # synthetic data, full tuning
    python -m cashflow_radar.train --source ibm    # the real IBM sample
    python -m cashflow_radar.train --fast          # 5 Optuna trials (CI)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime

import joblib
import mlflow
import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data.load import load_data
from .decision import compare_decision_rules, expected_net_value, global_threshold_curve, simulate_policies
from .explain import Explainer
from .features import FEATURES, build_features
from .modeling import (
    Calibrator,
    classification_metrics,
    days_late_regressor,
    logistic_baseline,
    regression_metrics,
    tune_xgb,
    xgb_classifier,
)
from .split import expanding_window_folds, time_split

log = logging.getLogger("cashflow_radar.train")


def predict_days_if_late(reg, X: pd.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(reg.predict(X)), 1.0, None)


def run(source: str | None = None, fast: bool = False) -> dict:
    cfg = load_config()
    seed = cfg["seed"]
    source = source or cfg["data"]["source"]
    cm, scfg = cfg["cost_model"], cfg["simulation"]
    art = resolve(cfg["paths"]["artifacts"]) / source
    art.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    inv, customers = load_data(source)
    split = time_split(inv, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    log.info("split: %s", split.summary())

    log.info("building point-in-time features for %d invoices", len(inv))
    feats = build_features(inv, inv, inv["invoice_date"], cfg["features"]["recent_window"])
    inv_i = inv.set_index("invoice_id")

    def xy(df):
        X = feats.loc[df["invoice_id"]]
        y = (inv_i.loc[df["invoice_id"], "days_late"] > 0).astype(int)
        return X, y.set_axis(X.index), inv_i.loc[df["invoice_id"]]

    Xtr, ytr, Itr = xy(split.train)
    Xva, yva, Iva = xy(split.valid)
    Xte, yte, Ite = xy(split.test)

    # --- baselines -----------------------------------------------------------------------
    metrics: dict = {"split": split.summary(), "late_rate": {
        "train": float(ytr.mean()), "valid": float(yva.mean()), "test": float(yte.mean())}}
    p_heur = Xte["hist_late_rate"].to_numpy()
    metrics["heuristic_customer_late_rate"] = classification_metrics(yte.to_numpy(), p_heur)
    lr = logistic_baseline(seed).fit(Xtr, ytr)
    metrics["logistic_regression"] = classification_metrics(yte.to_numpy(), lr.predict_proba(Xte)[:, 1])

    # --- tuned XGBoost -----------------------------------------------------------------
    n_trials = 5 if fast else cfg["tuning"]["n_trials"]
    folds = list(expanding_window_folds(split.train.set_index("invoice_id", drop=False),
                                        cfg["tuning"]["cv_folds"]))
    log.info("Optuna: %d trials x %d expanding-window folds", n_trials, len(folds))
    best = tune_xgb(Xtr, ytr, folds, n_trials, seed)
    clf = xgb_classifier(best, seed).fit(Xtr, ytr)
    p_va_raw = clf.predict_proba(Xva)[:, 1]
    p_te_raw = clf.predict_proba(Xte)[:, 1]
    metrics["xgboost_uncalibrated"] = classification_metrics(yte.to_numpy(), p_te_raw)

    cal = Calibrator().fit(p_va_raw, yva.to_numpy())
    p_va, p_te = cal.transform(p_va_raw), cal.transform(p_te_raw)
    metrics["xgboost_calibrated"] = classification_metrics(yte.to_numpy(), p_te)
    metrics["calibration_method"] = cal.method
    metrics["calibration_selection_log_loss"] = cal.selection_log_loss

    # --- days late (two-part model) ----------------------------------------------------
    late_tr = ytr == 1
    reg = days_late_regressor(seed).fit(Xtr[late_tr], np.log1p(Itr.loc[late_tr.to_numpy(), "days_late"]))
    days_if_late_te = predict_days_if_late(reg, Xte)
    exp_days_te = p_te * days_if_late_te
    naive_days = Xte["hist_mean_days_late"].fillna(Itr["days_late"].mean()).to_numpy()
    metrics["expected_days_late"] = {
        "model": regression_metrics(Ite["days_late"].to_numpy(), exp_days_te),
        "baseline_customer_mean": regression_metrics(Ite["days_late"].to_numpy(), naive_days),
    }

    # --- cost-based decisions (anything tuned is tuned on validation, reported on test) ---
    days_if_late_va = predict_days_if_late(reg, Xva)
    curve_va = global_threshold_curve(p_va, Iva["amount"].to_numpy(), Iva["days_late"].to_numpy(), cm)
    global_t = float(curve_va.loc[curve_va["net_savings"].idxmax(), "threshold"])
    metrics["decision_rules_test"] = compare_decision_rules(
        p_te, Ite["amount"].to_numpy(), Ite["days_late"].to_numpy(), days_if_late_te, global_t, cm)
    metrics["decision_rules_valid"] = compare_decision_rules(
        p_va, Iva["amount"].to_numpy(), Iva["days_late"].to_numpy(), days_if_late_va, global_t, cm)
    metrics["global_threshold_tuned_on_valid"] = global_t

    # --- DSO policy simulation on the test period ----------------------------------------
    scores_te = pd.DataFrame({"invoice_id": Xte.index, "p_late": p_te, "expected_net_value":
                              expected_net_value(p_te, Ite["amount"].to_numpy(), days_if_late_te, cm)})
    sim = simulate_policies(split.test, scores_te, customers.set_index("customer_id")["name"], cm,
                            scfg["capacity_share"])
    metrics["simulation"] = sim.to_dict(orient="records")

    # --- explanations -------------------------------------------------------------------
    expl = Explainer(clf)
    sample = Xte.sample(min(3000, len(Xte)), random_state=seed)
    importance = expl.global_importance(sample)
    metrics["shap_global_importance"] = importance.round(4).to_dict()

    # --- persist --------------------------------------------------------------------------
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    bundle = {"classifier": clf, "calibrator": cal, "regressor": reg, "global_threshold": global_t,
              "features": FEATURES, "cost_model": cm, "source": source, "version": version,
              "best_params": best, "recent_window": cfg["features"]["recent_window"]}
    joblib.dump(bundle, art / "model.joblib")
    reliability = _reliability_table(yte.to_numpy(), p_te_raw, p_te)
    reliability.to_csv(art / "reliability.csv", index=False)
    curve_va.to_csv(art / "threshold_curve_valid.csv", index=False)
    sim.to_csv(art / "policy_simulation.csv", index=False)
    metrics["model_version"] = version
    metrics["best_params"] = best
    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    _plots(art, reliability, curve_va, global_t, importance, sim)

    # --- MLflow -----------------------------------------------------------------------
    mlruns = resolve(cfg["paths"]["mlflow_uri"])
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlruns / 'mlflow.db'}")
    mlflow.set_experiment("cashflow-radar")
    with mlflow.start_run(run_name=f"{source}-{version}"):
        mlflow.log_params({"source": source, "n_trials": n_trials, **{f"xgb_{k}": v for k, v in best.items()},
                           **{f"cost_{k}": v for k, v in cm.items()}, "calibration": cal.method})
        for model_name in ["heuristic_customer_late_rate", "logistic_regression",
                           "xgboost_uncalibrated", "xgboost_calibrated"]:
            for k, v in metrics[model_name].items():
                mlflow.log_metric(f"{model_name}.{k}", v)
        mlflow.log_metric("global_threshold", global_t)
        for k, v in metrics["decision_rules_test"].items():
            if k.startswith("per-invoice"):
                mlflow.log_metric("test_net_savings_breakeven_rule", v["net_savings"])
        mlflow.log_metric("mae_expected_days_late", metrics["expected_days_late"]["model"]["mae_days_late"])
        mlflow.log_artifacts(str(art))
    log.info("done in %.0fs -> %s", time.time() - t0, art)
    return metrics


def _reliability_table(y, p_raw, p_cal, n_bins=10) -> pd.DataFrame:
    rows = []
    for name, p in [("uncalibrated", p_raw), ("calibrated", p_cal)]:
        b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
        for k in range(n_bins):
            m = b == k
            if m.sum() >= 10:
                rows.append({"model": name, "bin": k, "mean_predicted": p[m].mean(),
                             "observed_rate": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def _plots(art, reliability, curve, thr, importance, sim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"
    blue, orange = "#2563eb", "#d97706"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=grid, lw=1)
    for name, c in [("uncalibrated", orange), ("calibrated", blue)]:
        r = reliability[reliability.model == name]
        ax.plot(r.mean_predicted, r.observed_rate, marker="o", color=c, lw=2, label=name)
    ax.set(xlabel="Predicted probability of late payment", ylabel="Observed late rate (test)",
           xlim=(0, 1), ylim=(0, 1), title="Reliability on the test period")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(art / "reliability.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(curve.threshold, curve.net_savings / 1000, color=blue, lw=2)
    ax.axvline(thr, color=ink, lw=1, ls="--")
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.92, f"best global cut-off = {thr:.2f}", color=ink)
    ax.set(xlabel="Decision threshold on P(late)", ylabel="Net savings, validation (k€)",
           title="Cost-based threshold")
    fig.tight_layout(); fig.savefig(art / "threshold_curve.png", dpi=130); plt.close(fig)

    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(top.index, top.values, color=blue)
    ax.set(xlabel="Mean |SHAP| (log-odds)", title="What drives the late-payment risk")
    fig.tight_layout(); fig.savefig(art / "shap_importance.png", dpi=130); plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "ibm"])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    m = run(args.source, args.fast)
    print(json.dumps({k: m[k] for k in ["heuristic_customer_late_rate", "logistic_regression",
                                        "xgboost_calibrated", "expected_days_late", "decision_rules_test"]}, indent=2))
    print(pd.DataFrame(m["simulation"]).to_string(index=False))


if __name__ == "__main__":
    main()
