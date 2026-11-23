"""Turning probabilities into collections decisions, and measuring them in DSO.

Cost model (all numbers from config/cost_model)
-----------------------------------------------
benefit of touching an invoice that would be ``d`` days late:
    B(d) = effectiveness * amount * (cost_of_capital / 365 * d + bad_debt_rate)
cost of one proactive touch:
    C = cost_per_touch

Cost-based decision threshold
-----------------------------
Touch invoice i when its expected benefit beats the cost:
    P(late)_i * B(E[days late | late]_i)  >  C
equivalently P(late)_i > p*_i = C / B_i. The threshold is per invoice: a large invoice is
worth a touch at 20% risk, a small one not even at 90%. This is why calibration matters:
the rule compares a probability to a cost ratio, so the probability has to mean what it says.

For comparison we also fit the textbook version, one global cut-off on P(late), chosen to
maximise net savings on the validation period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def touch_benefit(amount: np.ndarray, days_late: np.ndarray, cm: dict) -> np.ndarray:
    """Realised value of touching an invoice that turned out ``days_late`` late (0 if on time)."""
    late = np.asarray(days_late) > 0
    return cm["touch_effectiveness"] * amount * (
        cm["cost_of_capital"] / 365.0 * days_late + cm["bad_debt_rate_if_late"] * late)


def benefit_if_late(amount: np.ndarray, exp_days_if_late: np.ndarray, cm: dict) -> np.ndarray:
    return cm["touch_effectiveness"] * amount * (
        cm["cost_of_capital"] / 365.0 * exp_days_if_late + cm["bad_debt_rate_if_late"])


def expected_net_value(p: np.ndarray, amount: np.ndarray, exp_days_if_late: np.ndarray, cm: dict) -> np.ndarray:
    """Ex-ante net value of a touch: P(late) x benefit if late - cost."""
    return p * benefit_if_late(amount, exp_days_if_late, cm) - cm["cost_per_touch"]


def breakeven_probability(amount: np.ndarray, exp_days_if_late: np.ndarray, cm: dict) -> np.ndarray:
    return np.minimum(1.0, cm["cost_per_touch"] / benefit_if_late(amount, exp_days_if_late, cm))


def realised_net_savings(act: np.ndarray, amount: np.ndarray, days_late: np.ndarray, cm: dict) -> float:
    act = np.asarray(act, dtype=bool)
    return float((touch_benefit(amount[act], days_late[act], cm) - cm["cost_per_touch"]).sum())


def global_threshold_curve(p: np.ndarray, amount: np.ndarray, days_late: np.ndarray, cm: dict) -> pd.DataFrame:
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    return pd.DataFrame({"threshold": grid,
                         "net_savings": [realised_net_savings(p >= t, amount, days_late, cm) for t in grid],
                         "touches": [int((p >= t).sum()) for t in grid]})


def compare_decision_rules(p, amount, days_late, exp_days_if_late, global_t, cm) -> dict:
    """Realised net savings of each rule on one period (no capacity limit)."""
    ev_act = expected_net_value(p, amount, exp_days_if_late, cm) > 0
    rules = {
        "touch nobody": np.zeros(len(p), bool),
        "touch everybody": np.ones(len(p), bool),
        "global cut-off P(late) >= 0.5": p >= 0.5,
        f"global cut-off P(late) >= {global_t:.2f} (tuned on validation)": p >= global_t,
        "per-invoice break-even (expected value > cost)": ev_act,
    }
    return {k: {"touches": int(a.sum()), "net_savings": realised_net_savings(a, amount, days_late, cm)}
            for k, a in rules.items()}


def simulate_policies(test: pd.DataFrame, scores: pd.DataFrame, customer_names: pd.Series,
                      cm: dict, capacity_share: float) -> pd.DataFrame:
    """Replay the test period under different prioritisation policies.

    Each week, collectors can proactively touch ``capacity_share`` of that week's new
    invoices. A touched invoice that would have been late is ``touch_effectiveness`` less
    late (fewer days and less write-off risk). Untouched invoices keep their real outcome.
    The model policy only spends a touch when its expected net value is positive.

    Metrics
      dso_days            amount-weighted average days from invoice to cash
      avg_overdue_balance average daily balance that is past due over the period
    """
    df = test[["invoice_id", "customer_id", "invoice_date", "amount", "days_late", "days_to_settle"]].copy()
    df = df.merge(scores[["invoice_id", "expected_net_value"]], on="invoice_id").reset_index(drop=True)
    df["customer_name"] = df["customer_id"].map(customer_names)
    df["week"] = df["invoice_date"].dt.to_period("W")
    period_days = max(1, (df["invoice_date"].max() - df["invoice_date"].min()).days)
    df["_true_value"] = touch_benefit(df["amount"].to_numpy(), df["days_late"].to_numpy(), cm)

    policies = {
        "no proactive touches": None,
        "alphabetical by customer": ("customer_name", True),
        "largest amount first": ("amount", False),
        "CashFlow Radar (model)": ("expected_net_value", False),
        "oracle (perfect foresight)": ("_true_value", False),
    }
    eff, amt = cm["touch_effectiveness"], df["amount"].to_numpy()
    rows = []
    for name, rule in policies.items():
        touched = np.zeros(len(df), dtype=bool)
        if rule is not None:
            col, asc = rule
            for _, g in df.groupby("week"):
                k = int(np.ceil(capacity_share * len(g)))
                if name.startswith("CashFlow"):
                    g = g[g["expected_net_value"] > 0]
                touched[g.sort_values(col, ascending=asc).index[:k]] = True
        new_late = np.where(touched, np.round(df["days_late"] * (1 - eff)), df["days_late"])
        new_settle = df["days_to_settle"] - (df["days_late"] - new_late)
        rows.append({
            "policy": name,
            "touches": int(touched.sum()),
            "dso_days": float((amt * new_settle).sum() / amt.sum()),
            "avg_overdue_balance": float((amt * new_late).sum() / period_days),
            "net_savings": float(df["_true_value"].to_numpy()[touched].sum() - cm["cost_per_touch"] * touched.sum()),
        })
    return pd.DataFrame(rows)
