"""Synthetic B2B invoice history in the canonical schema.

Why synthetic data at all? The IBM sample has 100 customers over ~2 years, which is too
small to show drift, seasonality and capacity-constrained collections. The generator
plants patterns a real AR ledger has, so the model has something real to find:

* a stable per-customer payment propensity (some customers are always late)
* a monthly liquidity cycle per customer (an AR(1) process): when cash is tight, every
  invoice paid in that stretch is later, and the customer's recent behaviour says so
* deteriorating customers whose delays grow month after month (and some that improve)
* disputes, which make an invoice much later
* paper invoices, longer payment terms and large invoices pay later
* December / year-end slowdown and a 4-month macro shock in 2023
Nothing in the model is told about these rules; it must recover them from history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, resolve
from .load import CUSTOMER_COLUMNS, INDUSTRIES, INVOICE_COLUMNS, SEGMENTS, company_name

COUNTRIES = ["DE", "AT", "CH", "NL", "FR", "IT", "ES", "PL", "SE", "GB"]
COUNTRY_EFFECT = {"DE": -0.2, "AT": -0.2, "CH": -0.3, "NL": -0.1, "SE": -0.3,
                  "FR": 0.2, "GB": 0.0, "PL": 0.2, "IT": 0.45, "ES": 0.4}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_customers: int | None = None, start: str | None = None, end: str | None = None,
             seed: int | None = None, save: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    scfg = cfg["data"]["synthetic"]
    n_customers = n_customers or scfg["n_customers"]
    start_ts, end_ts = pd.Timestamp(start or scfg["start"]), pd.Timestamp(end or scfg["end"])
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    horizon = (end_ts - start_ts).days

    shock_start, shock_end = pd.Timestamp("2023-03-01"), pd.Timestamp("2023-07-01")

    customers, invoices = [], []
    inv_counter = 100000
    for i in range(n_customers):
        cid = f"C{i:04d}"
        segment = rng.choice(SEGMENTS, p=[0.55, 0.32, 0.13])
        country = rng.choice(COUNTRIES)
        industry = rng.choice(INDUSTRIES)
        propensity = rng.normal(0, 1)                     # latent "how late does this customer pay"
        terms = int(rng.choice([30, 45, 60, 90], p=[0.45, 0.2, 0.25, 0.1]) if segment != "SMB"
                    else rng.choice([14, 30], p=[0.4, 0.6]))
        amount_mu = {"SMB": 7.0, "Mid-market": 8.6, "Enterprise": 10.0}[segment]
        rate_per_month = rng.gamma(2.0, {"SMB": 1.2, "Mid-market": 2.0, "Enterprise": 3.0}[segment])
        onboard = start_ts + pd.Timedelta(days=int(rng.integers(0, horizon * 0.5)))
        paperless_since = onboard + pd.Timedelta(days=int(rng.integers(0, 900))) if rng.random() < 0.7 else pd.NaT

        # Regime change: 15% deteriorate, 8% improve, starting at a random date.
        regime = rng.choice(["stable", "worse", "better"], p=[0.77, 0.15, 0.08])
        change_at = start_ts + pd.Timedelta(days=int(rng.integers(horizon * 0.3, horizon * 0.95)))
        ramp_days = int(rng.integers(90, 240))

        customers.append(dict(customer_id=cid, name=company_name(cid), industry=industry,
                              segment=segment, country=country))

        # Liquidity cycle: one AR(1) value per calendar month over the whole horizon.
        n_months = (end_ts.year - start_ts.year) * 12 + end_ts.month - start_ts.month + 1
        liquidity = np.zeros(n_months)
        for m in range(1, n_months):
            liquidity[m] = 0.85 * liquidity[m - 1] + rng.normal(0, 0.45)

        n = rng.poisson(rate_per_month * (end_ts - onboard).days / 30.4)
        if n == 0:
            continue
        offsets = np.sort(rng.integers(0, (end_ts - onboard).days, n))
        inv_dates = onboard + pd.to_timedelta(offsets, unit="D")
        amounts = np.round(rng.lognormal(amount_mu, 0.7, n), 2)
        rel_amount = np.log(amounts) - amount_mu

        t_since_change = (inv_dates - change_at).days.to_numpy()
        drift = np.clip(t_since_change / ramp_days, 0, 1) * (t_since_change > 0)
        drift = drift * (1.8 if regime == "worse" else -1.2 if regime == "better" else 0.0)

        paper = np.array([pd.isna(paperless_since) or d < paperless_since for d in inv_dates])
        disputed = rng.random(n) < _sigmoid(-3.2 + 0.5 * rel_amount + 0.25 * propensity)
        december = np.array([d.month == 12 for d in inv_dates])
        shock = np.array([shock_start <= d < shock_end for d in inv_dates])
        month_idx = ((inv_dates.year - start_ts.year) * 12 + inv_dates.month - start_ts.month).to_numpy()
        liq = liquidity[month_idx]

        logit = (-0.9 + 1.05 * propensity + drift + 1.8 * disputed + 0.35 * paper
                 + 0.35 * december + 0.6 * shock + 0.25 * rel_amount + 0.006 * (terms - 30)
                 + liq + COUNTRY_EFFECT[country] + rng.normal(0, 0.5, n))
        late = rng.random(n) < _sigmoid(logit)
        scale = np.exp(1.7 + 0.35 * propensity + 0.45 * np.clip(drift, 0, None) + 0.9 * disputed + 0.3 * shock
                       + 0.3 * np.clip(liq, 0, None))
        days_late = np.where(late, np.ceil(rng.gamma(1.4, scale)).astype(int), 0)
        days_late = np.minimum(days_late, 180)

        due = inv_dates + pd.Timedelta(days=terms)
        early = np.floor(terms * rng.uniform(0.35, 1.0, n)).astype(int)
        settle_offset = np.where(late, terms + days_late, early)
        settled = inv_dates + pd.to_timedelta(settle_offset, unit="D")

        for k in range(n):
            inv_counter += 1
            invoices.append((f"INV{inv_counter}", cid, country, inv_dates[k], due[k], settled[k],
                             float(amounts[k]), bool(disputed[k]), bool(not paper[k]),
                             int(settle_offset[k]), int(days_late[k])))

    inv = pd.DataFrame(invoices, columns=INVOICE_COLUMNS).sort_values(["invoice_date", "invoice_id"])
    inv = inv.reset_index(drop=True)
    cust = pd.DataFrame(customers)[CUSTOMER_COLUMNS]

    if save:
        out = resolve(scfg["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        inv.to_csv(out / "invoices.csv", index=False, date_format="%Y-%m-%d")
        cust.to_csv(out / "customers.csv", index=False)
    return inv, cust


if __name__ == "__main__":
    inv, cust = generate(save=True)
    print(f"{len(inv):,} invoices, {len(cust)} customers, late rate {inv.days_late.gt(0).mean():.1%}")
