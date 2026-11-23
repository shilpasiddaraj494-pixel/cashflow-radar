"""Point-in-time feature engineering.

The one rule: a feature for an invoice scored at time ``as_of`` may only use facts that
were known at ``as_of``. For customer payment history that means invoices *settled*
before ``as_of`` (not merely issued before it): an invoice issued last week whose payment
hasn't arrived yet has no label yet. Using it would leak the future into training.

Training rows are scored at their invoice date; the live worklist scores open invoices at
the snapshot date. Both go through the same function, so there is no train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "log_amount", "terms_days", "is_paperless", "is_december", "is_quarter_end",
    "hist_n_settled", "hist_late_rate", "hist_recent_late_rate", "hist_late_rate_90d",
    "hist_mean_days_late", "hist_max_days_late",
    "hist_recent_mean_days_late", "hist_trend_days_late", "hist_months_worsening",
    "hist_ewm_days_late", "hist_dispute_rate", "hist_settle_ratio", "days_since_last_late",
    "open_n", "open_amount_log", "open_overdue_n", "amount_vs_customer_mean", "is_new_customer",
]
CATEGORICAL_FEATURES = ["country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PRIOR_STRENGTH = 5.0  # pseudo-invoices pulling a short history toward the portfolio late rate


def _months_worsening(settled: np.ndarray, days_late: np.ndarray, as_of: pd.Timestamp) -> int:
    """How many months in a row the customer's average delay has grown (0-3).

    Looks at the last four complete calendar months before ``as_of``; a month with no
    settled invoices breaks the streak.
    """
    month0 = (as_of.to_period("M") - 1)
    means = []
    for k in range(3, -1, -1):  # oldest -> newest
        m = month0 - k
        mask = (settled >= m.start_time.to_datetime64()) & (settled <= m.end_time.to_datetime64())
        means.append(days_late[mask].mean() if mask.any() else np.nan)
    streak = 0
    for prev, cur in zip(means[-2::-1], means[:0:-1], strict=True):  # walk backwards from newest
        if np.isnan(prev) or np.isnan(cur) or cur <= prev:
            break
        streak += 1
    return streak


def build_features(history: pd.DataFrame, rows: pd.DataFrame, as_of: pd.Series | pd.Timestamp,
                   recent_window: int = 3) -> pd.DataFrame:
    """Compute features for ``rows`` (invoices to score) at time ``as_of``.

    history: every invoice we know about (settled_date may lie after as_of; those are
             treated as still open).
    rows:    invoices to score (subset of history or new invoices with the same columns,
             settled_date/days_late not required).
    as_of:   scoring time per row (Series aligned to rows) or a single timestamp.
    """
    rows = rows.reset_index(drop=True)
    as_of = (pd.Series(as_of, index=rows.index) if isinstance(as_of, pd.Timestamp)
             else pd.Series(pd.to_datetime(as_of).to_numpy(), index=rows.index))

    g_settled = np.sort(history["settled_date"].to_numpy())
    g_late_cum = np.concatenate([[0], np.cumsum(
        history.sort_values("settled_date")["days_late"].to_numpy() > 0)])

    out = np.full((len(rows), len(NUMERIC_FEATURES)), np.nan)
    col = {f: i for i, f in enumerate(NUMERIC_FEATURES)}
    hist_by_cust = {c: g for c, g in history.groupby("customer_id", sort=False)}

    for cust, idx in rows.groupby("customer_id", sort=False).groups.items():
        h = hist_by_cust.get(cust)
        if h is not None:
            h = h.sort_values("settled_date")
            h_inv = h["invoice_date"].to_numpy()
            h_due = h["due_date"].to_numpy()
            h_set = h["settled_date"].to_numpy()
            h_late = h["days_late"].to_numpy().astype(float)
            h_amt = h["amount"].to_numpy()
            h_disp = h["disputed"].to_numpy().astype(float)
            h_ratio = (h["days_to_settle"] / (h["due_date"] - h["invoice_date"]).dt.days.clip(lower=1)).to_numpy()
            h_ids = h["invoice_id"].to_numpy()
        for r in idx:
            t = as_of[r]
            t64 = t.to_datetime64()
            f = out[r]
            if h is None:
                n = 0
            else:
                n = int(np.searchsorted(h_set, t64, side="left"))  # settled strictly before t
            g_n = int(np.searchsorted(g_settled, t64, side="left"))
            prior = g_late_cum[g_n] / g_n if g_n else 0.4
            f[col["hist_n_settled"]] = n
            f[col["is_new_customer"]] = float(n == 0)
            if n:
                dl = h_late[:n]
                late_flags = dl > 0
                f[col["hist_late_rate"]] = (late_flags.sum() + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH)
                f[col["hist_mean_days_late"]] = dl.mean()
                f[col["hist_max_days_late"]] = dl.max()
                f[col["hist_recent_late_rate"]] = (late_flags[-10:].sum() + 2 * prior) / (min(n, 10) + 2)
                in_90d = h_set[:n] >= t64 - np.timedelta64(90, "D")
                f[col["hist_late_rate_90d"]] = (late_flags[in_90d].sum() + 2 * prior) / (in_90d.sum() + 2)
                recent = dl[-recent_window:].mean()
                f[col["hist_recent_mean_days_late"]] = recent
                f[col["hist_trend_days_late"]] = recent - dl.mean()
                w = 0.5 ** (np.arange(n)[::-1] / 5.0)  # half-life of 5 invoices
                f[col["hist_ewm_days_late"]] = float((w * dl).sum() / w.sum())
                f[col["hist_dispute_rate"]] = h_disp[:n].mean()
                f[col["hist_settle_ratio"]] = h_ratio[:n].mean()
                last_late = h_set[:n][late_flags]
                f[col["days_since_last_late"]] = ((t64 - last_late.max()) / np.timedelta64(1, "D")
                                                  if late_flags.any() else 9999.0)
                f[col["hist_months_worsening"]] = _months_worsening(h_set[:n], dl, t)
            else:
                f[col["hist_late_rate"]] = prior
                f[col["hist_recent_late_rate"]] = prior
                f[col["hist_late_rate_90d"]] = prior
                f[col["hist_months_worsening"]] = 0
                f[col["days_since_last_late"]] = 9999.0
            if h is not None:
                issued_before = h_inv < t64
                open_mask = issued_before & (h_set >= t64)
                # exclude the row itself when it is part of history
                not_self = h_ids != rows.at[r, "invoice_id"]
                open_mask &= not_self
                f[col["open_n"]] = open_mask.sum()
                f[col["open_amount_log"]] = np.log1p(h_amt[open_mask].sum())
                f[col["open_overdue_n"]] = (open_mask & (h_due < t64)).sum()
                prev_amt = h_amt[issued_before & not_self]
                f[col["amount_vs_customer_mean"]] = (rows.at[r, "amount"] / prev_amt.mean()
                                                     if len(prev_amt) else 1.0)
            else:
                f[col["open_n"]] = 0
                f[col["open_amount_log"]] = 0
                f[col["open_overdue_n"]] = 0
                f[col["amount_vs_customer_mean"]] = 1.0

    feats = pd.DataFrame(out, columns=NUMERIC_FEATURES)
    feats["log_amount"] = np.log1p(rows["amount"].to_numpy())
    feats["terms_days"] = (rows["due_date"] - rows["invoice_date"]).dt.days.to_numpy()
    feats["is_paperless"] = rows["is_paperless"].astype(float).to_numpy()
    # Customer tenure is deliberately not a feature either: over a short history it is
    # mostly a proxy for calendar time.
    # Month-of-year is deliberately not a feature: with only a few years of history it
    # memorises one-off periods (e.g. a 2023 shock) instead of a repeatable season.
    feats["is_december"] = rows["invoice_date"].dt.month.eq(12).astype(float).to_numpy()
    feats["is_quarter_end"] = rows["invoice_date"].dt.month.isin([3, 6, 9, 12]).astype(float).to_numpy()
    feats["country"] = rows["country"].astype(str).to_numpy()
    feats.index = rows["invoice_id"].to_numpy()
    return feats[FEATURES]
