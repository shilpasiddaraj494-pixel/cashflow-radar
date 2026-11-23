"""CashFlow Radar API.

    uvicorn cashflow_radar.api.main:app --reload
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from ..db import get_engine
from ..explain import Explainer
from ..score import load_bundle, score_frame
from .schemas import InvoiceDetail, ScoreRequest, ScoreResponse, Summary, Worklist, WorklistItem

STATE: dict = {}

WORKLIST_SQL = """
SELECT s.priority_rank, s.invoice_id, i.customer_id, c.name AS customer_name, i.country, c.segment,
       c.industry, i.amount, i.invoice_date, i.due_date, i.status, s.p_late, s.expected_days_late,
       s.expected_net_value, s.breakeven_p, s.action, s.reasons, s.as_of
FROM invoice_scores s
JOIN invoices i ON i.invoice_id = s.invoice_id
JOIN customers c ON c.customer_id = i.customer_id
"""


def _info() -> dict:
    if "info" not in STATE:
        with get_engine().connect() as conn:
            row = conn.execute(text("SELECT * FROM model_info")).mappings().first()
        if row is None:
            raise HTTPException(503, "no worklist loaded; run `python -m cashflow_radar.score`")
        STATE["info"] = dict(row)
    return STATE["info"]


def _to_item(r: dict) -> dict:
    as_of = pd.Timestamp(r["as_of"])
    r = dict(r)
    r["reasons"] = json.loads(r["reasons"])
    r["invoice_date"] = pd.Timestamp(r["invoice_date"]).date()
    r["due_date"] = pd.Timestamp(r["due_date"]).date()
    r["days_to_due"] = (pd.Timestamp(r["due_date"]) - as_of).days
    return r


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.clear()
    yield


app = FastAPI(title="CashFlow Radar", version="1.0", lifespan=lifespan,
              description="Late-payment risk scores and a ranked collections worklist.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/api/summary", response_model=Summary)
def summary():
    info = _info()
    metrics = json.loads(info["metrics"])
    with get_engine().connect() as conn:
        df = pd.read_sql(text(WORKLIST_SQL), conn)
    overdue = df["status"] == "overdue"
    touch = df["action"] == "touch"
    return Summary(
        as_of=pd.Timestamp(info["as_of"]).date(), model_version=info["model_version"], source=info["source"],
        open_invoices=len(df), open_balance=float(df["amount"].sum()),
        overdue_balance=float(df.loc[overdue, "amount"].sum()),
        expected_late_balance=float((df["amount"] * df["p_late"]).sum()),
        recommended_touches=int(touch.sum()),
        recommended_touch_value=float(df.loc[touch, "expected_net_value"].sum()),
        test_metrics={k: metrics[k] for k in ["heuristic_customer_late_rate", "logistic_regression",
                                              "xgboost_uncalibrated", "xgboost_calibrated"]},
        policy_simulation=metrics["simulation"],
        cost_model=json.loads(info["cost_model"]),
        capacity_share=float(info["capacity_share"]),
    )


@app.get("/api/worklist", response_model=Worklist)
def worklist(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
             action: str = Query("all", pattern="^(all|touch|monitor)$"),
             status: str = Query("all", pattern="^(all|open|overdue)$"),
             q: str | None = Query(None, description="search customer name or invoice id")):
    info = _info()
    where, params = [], {}
    if action != "all":
        where.append("s.action = :action"); params["action"] = action
    if status != "all":
        where.append("i.status = :status"); params["status"] = status
    if q:
        where.append("(LOWER(c.name) LIKE :q OR LOWER(s.invoice_id) LIKE :q)"); params["q"] = f"%{q.lower()}%"
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    with get_engine().connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM ({WORKLIST_SQL}{clause}) t"), params).scalar_one()
        rows = conn.execute(text(f"{WORKLIST_SQL}{clause} ORDER BY s.priority_rank LIMIT :limit OFFSET :offset"),
                            {**params, "limit": limit, "offset": offset}).mappings().all()
    return Worklist(as_of=pd.Timestamp(info["as_of"]).date(), total=total,
                    items=[WorklistItem(**_to_item(r)) for r in rows])


@app.get("/api/invoices/{invoice_id}", response_model=InvoiceDetail)
def invoice(invoice_id: str):
    with get_engine().connect() as conn:
        r = conn.execute(text(f"{WORKLIST_SQL} WHERE s.invoice_id = :id"), {"id": invoice_id}).mappings().first()
        if r is None:
            raise HTTPException(404, "invoice not on the worklist")
        hist = pd.read_sql(text(
            "SELECT invoice_id, invoice_date, settled_date, amount, days_late FROM invoices "
            "WHERE customer_id = :c AND status = 'settled' ORDER BY settled_date DESC LIMIT 24"),
            conn, params={"c": r["customer_id"]})
    hist = hist.iloc[::-1]
    for c in ["invoice_date", "settled_date"]:
        hist[c] = pd.to_datetime(hist[c]).dt.date
    hist["days_late"] = hist["days_late"].astype(int)
    return InvoiceDetail(**_to_item(r), history=hist.to_dict(orient="records"))


@app.post("/api/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Score a new (not yet issued or just issued) invoice for an existing customer."""
    info = _info()
    if "bundle" not in STATE:
        STATE["bundle"] = load_bundle(info["source"])
        STATE["explainer"] = Explainer(STATE["bundle"]["classifier"])
        with get_engine().connect() as conn:
            h = pd.read_sql(text("SELECT * FROM invoices"), conn)
            cust = pd.read_sql(text("SELECT customer_id, country FROM customers"), conn)
        for c in ["invoice_date", "due_date", "settled_date"]:
            h[c] = pd.to_datetime(h[c])
        # open invoices have no settlement yet: push it past any scoring date
        h["settled_date"] = h["settled_date"].fillna(pd.Timestamp("2262-01-01"))
        h["days_late"] = h["days_late"].fillna(0)
        h["days_to_settle"] = h["days_to_settle"].fillna(0)
        h["disputed"] = h["disputed"].astype(bool)
        STATE["history"], STATE["countries"] = h, cust.set_index("customer_id")["country"]
    countries = STATE["countries"]
    if req.customer_id not in countries.index:
        raise HTTPException(404, f"unknown customer {req.customer_id}")
    row = pd.DataFrame([{
        "invoice_id": "__new__", "customer_id": req.customer_id,
        "country": req.country or countries[req.customer_id], "amount": req.amount,
        "invoice_date": pd.Timestamp(req.invoice_date), "due_date": pd.Timestamp(req.due_date),
        "is_paperless": req.is_paperless,
    }])
    as_of = max(pd.Timestamp(req.invoice_date), pd.Timestamp(info["as_of"]))
    s = score_frame(STATE["bundle"], STATE["history"], row, as_of, STATE["explainer"]).iloc[0]
    return ScoreResponse(customer_id=req.customer_id, p_late=s.p_late, expected_days_late=s.expected_days_late,
                         expected_days_late_if_late=s.expected_days_late_if_late,
                         expected_net_value=s.expected_net_value, breakeven_p=s.breakeven_p, action=s.action,
                         reasons=json.loads(s.reasons), model_version=STATE["bundle"]["version"])


@app.get("/api/customers")
def customers(limit: int = 500) -> list[dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT customer_id, name, country, segment FROM customers ORDER BY name LIMIT :l"),
                            {"l": limit}).mappings()
        return [dict(r) for r in rows]


