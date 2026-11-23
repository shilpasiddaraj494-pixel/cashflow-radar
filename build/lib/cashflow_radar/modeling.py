"""Models: a heuristic baseline, logistic regression, and tuned + calibrated XGBoost.

Two-part model for "how late":
    P(late)                  - calibrated XGBoost classifier
    E[days late | late]      - XGBoost regressor trained on late invoices only (log target)
    expected days late       = P(late) * E[days late | late]
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _onehot():
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)


def tree_preprocessor() -> ColumnTransformer:
    # Trees handle NaN natively; only the country needs encoding.
    return ColumnTransformer([("num", "passthrough", NUMERIC_FEATURES),
                              ("cat", _onehot(), CATEGORICAL_FEATURES)], verbose_feature_names_out=False)


def logistic_baseline(seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", _onehot(), CATEGORICAL_FEATURES)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=0.5, random_state=seed))])


def xgb_classifier(params: dict, seed: int) -> Pipeline:
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                            random_state=seed, n_jobs=4, **params)
    return Pipeline([("pre", tree_preprocessor()), ("clf", clf)])


def tune_xgb(X: pd.DataFrame, y: pd.Series, folds: list, n_trials: int, seed: int) -> dict:
    """Optuna search; each trial is scored by mean log loss over expanding-window folds."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        }
        losses = []
        for tr, va in folds:
            m = xgb_classifier(params, seed).fit(X.loc[tr], y.loc[tr])
            losses.append(log_loss(y.loc[va], m.predict_proba(X.loc[va])[:, 1], labels=[0, 1]))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


class Calibrator:
    """Pick none / Platt / isotonic out-of-time, then refit the winner on all validation data.

    Calibration fitted on one period can make things worse on the next one if the base
    rate moves. So the method is chosen by fitting on the first half of the validation
    period and scoring log loss on the second half; "none" wins if nothing helps.
    """

    METHODS = ("none", "platt", "isotonic")

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def _fit_method(self, method, p, y):
        if method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        if method == "platt":
            return LogisticRegression(C=1e6).fit(self._logit(p), y)
        return None

    def _apply(self, method, model, p):
        if method == "isotonic":
            return model.predict(p)
        if method == "platt":
            return model.predict_proba(self._logit(p))[:, 1]
        return p

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        """p and y must be in chronological order."""
        half = len(p) // 2
        self.selection_log_loss = {}
        for m in self.METHODS:
            if m == "isotonic" and half < 500:
                continue
            fitted = self._fit_method(m, p[:half], y[:half])
            pred = np.clip(self._apply(m, fitted, p[half:]), 1e-6, 1 - 1e-6)
            self.selection_log_loss[m] = float(log_loss(y[half:], pred, labels=[0, 1]))
        self.method = min(self.selection_log_loss, key=self.selection_log_loss.get)
        self.model = self._fit_method(self.method, p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self._apply(self.method, self.model, p)


def days_late_regressor(seed: int) -> Pipeline:
    reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.05,
                           max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                           tree_method="hist", random_state=seed, n_jobs=4)
    return Pipeline([("pre", tree_preprocessor()), ("reg", reg)])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
    }


def regression_metrics(days_late: np.ndarray, expected: np.ndarray) -> dict:
    return {"mae_days_late": float(mean_absolute_error(days_late, expected))}
