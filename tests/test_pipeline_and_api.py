"""End-to-end: train (fast) on a small synthetic book, score a worklist, hit the API."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def trained():
    from cashflow_radar import score, train
    metrics = train.run("synthetic", fast=True)
    summary = score.run("synthetic")
    return metrics, summary


@pytest.fixture(scope="module")
def client(trained):
    from cashflow_radar.api.main import app
    return TestClient(app)


def test_training_produces_sane_metrics(trained):
    m, _ = trained
    xgb = m["xgboost_calibrated"]
    assert 0.6 < xgb["roc_auc"] <= 1
    assert xgb["ece"] < 0.15
    rules = m["decision_rules_test"]
    assert rules["touch nobody"]["net_savings"] == 0
    sim = {r["policy"]: r for r in m["simulation"]}
    oracle, none = sim["oracle (perfect foresight)"], sim["no proactive touches"]
    assert oracle["avg_overdue_balance"] <= none["avg_overdue_balance"]


def test_worklist_is_ranked_and_explained(client, trained):
    _, s = trained
    r = client.get("/api/worklist", params={"limit": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == s["open_invoices"]
    ranks = [i["priority_rank"] for i in body["items"]]
    assert ranks == sorted(ranks)
    values = [i["expected_net_value"] for i in body["items"]]
    assert values == sorted(values, reverse=True)
    assert all(1 <= len(i["reasons"]) <= 6 for i in body["items"])
    assert all(i["shap_base"] is not None for i in body["items"])


def test_overdue_invoices_are_certainly_late(client):
    items = client.get("/api/worklist", params={"status": "overdue", "limit": 50}).json()["items"]
    assert items, "expected some overdue invoices in the snapshot"
    assert all(i["p_late"] == 1.0 and i["days_to_due"] < 0 for i in items)
    assert all(i["reasons"][0]["feature"] == "days_overdue" for i in items)


def test_filters_and_detail(client):
    touch = client.get("/api/worklist", params={"action": "touch", "limit": 5}).json()
    assert all(i["action"] == "touch" and i["expected_net_value"] > 0 for i in touch["items"])
    inv_id = touch["items"][0]["invoice_id"]
    d = client.get(f"/api/invoices/{inv_id}").json()
    assert d["invoice_id"] == inv_id
    assert all(h["days_late"] >= 0 for h in d["history"])
    assert client.get("/api/invoices/does-not-exist").status_code == 404


def test_summary_and_score_endpoint(client):
    s = client.get("/api/summary").json()
    assert s["open_balance"] >= s["overdue_balance"] > 0
    cust = client.get("/api/customers").json()[0]["customer_id"]
    req = {"customer_id": cust, "amount": 25_000, "invoice_date": s["as_of"], "due_date": s["as_of"]}
    r = client.post("/api/score", json=req)
    assert r.status_code == 200
    out = r.json()
    assert 0 <= out["p_late"] <= 1 and out["action"] in {"touch", "monitor"}
    bad = client.post("/api/score", json={**req, "due_date": "2000-01-01"})
    assert bad.status_code == 422


def test_sorting_is_whitelisted(client):
    items = client.get("/api/worklist", params={"sort": "amount", "desc": True, "limit": 30}).json()["items"]
    amounts = [i["amount"] for i in items]
    assert amounts == sorted(amounts, reverse=True)
    assert client.get("/api/worklist", params={"sort": "amount; DROP TABLE invoices"}).status_code == 422
