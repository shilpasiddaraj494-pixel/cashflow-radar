"""Batch scoring: build today's collections worklist and load it into the database.

    python -m cashflow_radar.score                       # synthetic, snapshot 60 days before data end
    python -m cashflow_radar.score --as-of 2024-10-15

What the database receives is only what was knowable at the snapshot date: open invoices
have no settlement date or days late, so the UI cannot leak the future either.
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data.load import load_data
from .db import get_engine
from .decision import breakeven_probability, expected_net_value
from .explain import Explainer
from .features import build_features

log = logging.getLogger("cashflow_radar.score")


def load_bundle(source: str) -> dict:
    return joblib.load(resolve(load_config()["paths"]["artifacts"]) / source / "model.joblib")


def score_frame(bundle: dict, history: pd.DataFrame, rows: pd.DataFrame, as_of, explainer: Explainer | None = None,
                top_k: int = 4) -> pd.DataFrame:
    """Score ``rows`` at ``as_of`` with the trained bundle. Returns one row per invoice."""
    X = build_features(history, rows, as_of, bundle["recent_window"])
    p_raw = bundle["classifier"].predict_proba(X)[:, 1]
    p = bundle["calibrator"].transform(p_raw)
    days_if_late = np.clip(np.expm1(bundle["regressor"].predict(X)), 1.0, None)
    # An open invoice already past its due date *is* late; the model (trained at issue time)
    # can't know that, so we condition on it: P(late) = 1 and at least the delay so far.
    as_of_ts = pd.to_datetime(pd.Series(as_of, index=rows.index) if not isinstance(as_of, pd.Series) else as_of)
    overdue_days = (as_of_ts.to_numpy() - rows["due_date"].to_numpy()) / np.timedelta64(1, "D")
    overdue = overdue_days > 0
    p = np.where(overdue, 1.0, p)
    days_if_late = np.where(overdue, np.maximum(days_if_late, overdue_days + 1), days_if_late)
    amount = rows["amount"].to_numpy()
    cm = bundle["cost_model"]
    env = expected_net_value(p, amount, days_if_late, cm)
    out = pd.DataFrame({
        "invoice_id": rows["invoice_id"].to_numpy(),
        "p_late": p,
        "expected_days_late_if_late": days_if_late,
        "expected_days_late": p * days_if_late,
        "expected_net_value": env,
        "breakeven_p": breakeven_probability(amount, days_if_late, cm),
        "action": np.where(env > 0, "touch", "monitor"),
    })
    explainer = explainer or Explainer(bundle["classifier"])
    reasons = explainer.reasons(X, top_k)
    for i in np.flatnonzero(overdue):
        d = int(overdue_days[i])
        reasons[i] = [{"feature": "days_overdue", "impact": 0.0, "direction": "raises risk",
                       "text": f"Already {d} day{'s' if d != 1 else ''} past due"}] + reasons[i][: top_k - 1]
    out["reasons"] = [json.dumps(r) for r in reasons]
    return out


def run(source: str | None = None, as_of: str | None = None) -> dict:
    cfg = load_config()
    source = source or cfg["data"]["source"]
    bundle = load_bundle(source)
    inv, customers = load_data(source)
    snapshot = pd.Timestamp(as_of) if as_of else (inv["invoice_date"].max() - pd.Timedelta(days=60)).normalize()

    known = inv[inv["invoice_date"] <= snapshot].copy()
    open_mask = known["settled_date"] > snapshot
    open_inv = known[open_mask].reset_index(drop=True)
    log.info("snapshot %s: %d open invoices", snapshot.date(), len(open_inv))

    scores = score_frame(bundle, known, open_inv, snapshot)
    scores["priority_rank"] = scores["expected_net_value"].rank(ascending=False, method="first").astype(int)
    scores["as_of"] = snapshot
    scores["model_version"] = bundle["version"]

    # Hide the future: open invoices lose their outcome columns.
    known.loc[open_mask, ["settled_date", "days_to_settle", "days_late"]] = None
    known["status"] = np.where(open_mask, np.where(known["due_date"] < snapshot, "overdue", "open"), "settled")

    metrics = json.loads((resolve(cfg["paths"]["artifacts"]) / source / "metrics.json").read_text())
    info = pd.DataFrame([{
        "model_version": bundle["version"], "source": source, "as_of": snapshot,
        "capacity_share": cfg["simulation"]["capacity_share"],
        "cost_model": json.dumps(bundle["cost_model"]), "metrics": json.dumps(metrics),
    }])

    eng = get_engine()
    with eng.begin() as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        known.to_sql("invoices", conn, if_exists="replace", index=False, chunksize=5000)
        scores.to_sql("invoice_scores", conn, if_exists="replace", index=False, chunksize=5000)
        info.to_sql("model_info", conn, if_exists="replace", index=False)
    summary = {"as_of": str(snapshot.date()), "open_invoices": len(open_inv),
               "recommended_touches": int((scores["action"] == "touch").sum())}
    log.info("loaded worklist: %s", summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "ibm"])
    ap.add_argument("--as-of")
    args = ap.parse_args()
    print(run(args.source, args.as_of))


if __name__ == "__main__":
    main()
