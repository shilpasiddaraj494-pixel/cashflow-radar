"""Export the dashboard as ONE html file that opens straight from disk (no server).

    cd frontend && npm run build:static && cd ..
    python scripts/export_static.py                    # 6 weekly snapshots, synthetic data
    python scripts/export_static.py --weeks 8 --source ibm

Each week is scored exactly as the live worklist would have been on that date: only
invoices open at that date, only payment history known by then.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cashflow_radar.config import load_config, resolve  # noqa: E402
from cashflow_radar.data.load import load_data  # noqa: E402
from cashflow_radar.explain import Explainer  # noqa: E402
from cashflow_radar.score import load_bundle, score_frame  # noqa: E402


def snapshot(bundle, explainer, inv, cust, as_of: pd.Timestamp) -> tuple[dict, list[dict]]:
    known = inv[inv["invoice_date"] <= as_of]
    open_inv = known[known["settled_date"] > as_of].reset_index(drop=True)
    s = score_frame(bundle, known, open_inv, as_of, explainer)
    s["priority_rank"] = s["expected_net_value"].rank(ascending=False, method="first").astype(int)
    df = open_inv.merge(s, on="invoice_id")
    df["status"] = np.where(df["due_date"] < as_of, "overdue", "open")
    df["days_to_due"] = (df["due_date"] - as_of).dt.days
    df = df.sort_values("priority_rank")
    items = [{
        "priority_rank": int(r.priority_rank), "invoice_id": r.invoice_id, "customer_id": r.customer_id,
        "amount": round(float(r.amount), 2), "invoice_date": str(r.invoice_date.date()),
        "due_date": str(r.due_date.date()), "status": r.status, "days_to_due": int(r.days_to_due),
        "p_late": round(float(r.p_late), 4), "expected_days_late": round(float(r.expected_days_late), 2),
        "expected_net_value": round(float(r.expected_net_value), 2), "breakeven_p": round(float(r.breakeven_p), 4),
        "action": r.action,
        # compact: [feature, impact, text]; the page expands it
        "reasons": [[x["feature"], x["impact"], x["text"]] for x in json.loads(r.reasons)],
        "shap_base": round(float(r.shap_base), 3), "shap_score": round(float(r.shap_score), 3),
    } for r in df.itertuples()]
    overdue = df["status"] == "overdue"
    touch = df["action"] == "touch"
    summary = {
        "as_of": str(as_of.date()), "open_invoices": len(df), "open_balance": float(df["amount"].sum()),
        "overdue_balance": float(df.loc[overdue, "amount"].sum()),
        "expected_late_balance": float((df["amount"] * df["p_late"]).sum()),
        "recommended_touches": int(touch.sum()),
        "recommended_touch_value": float(df.loc[touch, "expected_net_value"].sum()),
    }
    return summary, items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, choices=["synthetic", "ibm"])
    ap.add_argument("--weeks", type=int, default=6)
    ap.add_argument("--out", default="cashflow-radar-dashboard.html")
    args = ap.parse_args()

    cfg = load_config()
    source = args.source or cfg["data"]["source"]
    bundle = load_bundle(source)
    explainer = Explainer(bundle["classifier"])
    inv, cust = load_data(source)
    metrics = json.loads((resolve(cfg["paths"]["artifacts"]) / source / "metrics.json").read_text())

    last = (inv["invoice_date"].max() - pd.Timedelta(days=60)).normalize()
    dates = [last - pd.Timedelta(weeks=k) for k in range(args.weeks)][::-1]
    weeks = []
    for d in dates:
        summary, items = snapshot(bundle, explainer, inv, cust, d)
        weeks.append({"summary": summary, "items": items})
        print(f"  {d.date()}: {len(items)} open invoices")

    common = {
        "model_version": bundle["version"], "source": source,
        "test_metrics": {k: metrics[k] for k in ["heuristic_customer_late_rate", "logistic_regression",
                                                 "xgboost_uncalibrated", "xgboost_calibrated"]},
        "policy_simulation": metrics["simulation"], "cost_model": bundle["cost_model"],
        "capacity_share": cfg["simulation"]["capacity_share"],
    }
    # Settled history per customer, enough to show the last 24 before the earliest week.
    first = dates[0]
    settled = inv[inv["settled_date"] <= dates[-1]].sort_values("settled_date")
    before = settled["settled_date"] < first
    from_end = settled[before].groupby("customer_id").cumcount(ascending=False)
    settled = settled[~before | before & (from_end.reindex(settled.index) < 24)]
    histories = {cid: [{"invoice_id": r.invoice_id, "invoice_date": str(r.invoice_date.date()),
                        "settled_date": str(r.settled_date.date()), "amount": round(float(r.amount), 2),
                        "days_late": int(r.days_late)} for r in g.itertuples()]
                 for cid, g in settled.groupby("customer_id")}
    details = {r.customer_id: {"customer_name": r.name, "industry": r.industry, "segment": r.segment,
                               "country": r.country} for r in cust.itertuples()}

    data = json.dumps({"common": common, "weeks": weeks, "histories": histories, "details": details},
                      separators=(",", ":")).replace("</", "<\\/")
    html = (ROOT / "frontend" / "dist-static" / "index.html").read_text(encoding="utf-8")
    html = html.replace("<head>", f"<head>\n<script>window.__CFR_DATA__={data};</script>", 1)
    (ROOT / args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out}: {len(weeks)} weeks, {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
