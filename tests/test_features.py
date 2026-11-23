import numpy as np
import pandas as pd

from cashflow_radar.data.synthetic import generate
from cashflow_radar.features import FEATURES, build_features


def _inv(rows):
    df = pd.DataFrame(rows, columns=["invoice_id", "invoice_date", "due_date", "settled_date", "amount"])
    for c in ["invoice_date", "due_date", "settled_date"]:
        df[c] = pd.to_datetime(df[c])
    df["customer_id"] = "C1"
    df["country"] = "DE"
    df["disputed"] = False
    df["is_paperless"] = True
    df["days_to_settle"] = (df["settled_date"] - df["invoice_date"]).dt.days
    df["days_late"] = (df["settled_date"] - df["due_date"]).dt.days.clip(lower=0)
    return df


def test_no_leakage_from_invoices_settled_after_as_of():
    """Changing the outcome of anything not yet settled at as_of must not move any feature."""
    inv, _ = generate(n_customers=12, seed=3)
    as_of = pd.Timestamp("2023-01-15")
    rows = inv[inv["invoice_date"] <= as_of].tail(40)
    before = build_features(inv, rows, as_of)

    tampered = inv.copy()
    future = tampered["settled_date"] >= as_of
    tampered.loc[future, "settled_date"] += pd.Timedelta(days=90)
    tampered.loc[future, "days_late"] += 90
    tampered.loc[future, "days_to_settle"] += 90
    tampered.loc[future, "disputed"] = True
    after = build_features(tampered, rows, as_of)
    pd.testing.assert_frame_equal(before, after)


def test_history_features_by_hand():
    inv = _inv([
        ("A", "2024-01-01", "2024-01-31", "2024-01-31", 100.0),  # on time
        ("B", "2024-02-01", "2024-03-02", "2024-03-12", 100.0),  # 10 days late
        ("C", "2024-03-01", "2024-03-31", "2024-05-30", 300.0),  # settled after as_of -> open
        ("D", "2024-04-01", "2024-05-01", "2024-05-10", 200.0),  # the invoice being scored
    ])
    f = build_features(inv, inv[inv.invoice_id == "D"], pd.Series([pd.Timestamp("2024-04-01")])).iloc[0]
    assert f["hist_n_settled"] == 2
    assert f["hist_mean_days_late"] == 5
    assert f["hist_max_days_late"] == 10
    assert f["open_n"] == 1                       # C is open, D itself is excluded
    assert f["open_overdue_n"] == 1               # C was due 31 Mar
    assert np.isclose(f["amount_vs_customer_mean"], 200 / np.mean([100, 100, 300]))
    assert f["is_new_customer"] == 0


def test_new_customer_gets_portfolio_prior():
    inv = _inv([("A", "2024-01-01", "2024-01-31", "2024-02-20", 100.0)])
    f = build_features(inv, inv, pd.Series([pd.Timestamp("2024-01-01")])).iloc[0]
    assert f["is_new_customer"] == 1
    assert np.isnan(f["hist_mean_days_late"])
    assert 0 < f["hist_late_rate"] < 1
    assert list(build_features(inv, inv, pd.Timestamp("2024-01-01")).columns) == FEATURES


def test_months_worsening_counts_consecutive_increases():
    rows = []
    for k, late in enumerate([1, 4, 9, 15]):  # Jan..Apr settlements, growing delays
        m = k + 1
        rows.append((f"I{m}", f"2024-{m:02d}-01", f"2024-{m:02d}-02", f"2024-{m:02d}-{2 + late:02d}", 100.0))
    rows.append(("NEW", "2024-05-03", "2024-06-02", "2024-06-10", 100.0))
    inv = _inv(rows)
    f = build_features(inv, inv[inv.invoice_id == "NEW"], pd.Timestamp("2024-05-03")).iloc[0]
    assert f["hist_months_worsening"] == 3
