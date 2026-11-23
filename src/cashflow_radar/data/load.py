"""Load invoices into one canonical schema, whatever the source.

Canonical invoice columns
-------------------------
invoice_id, customer_id, country, invoice_date, due_date, settled_date, amount,
disputed, is_paperless, days_to_settle, days_late

Canonical customer columns
--------------------------
customer_id, name, industry, segment, country
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..config import load_config, resolve

INVOICE_COLUMNS = [
    "invoice_id", "customer_id", "country", "invoice_date", "due_date", "settled_date",
    "amount", "disputed", "is_paperless", "days_to_settle", "days_late",
]
CUSTOMER_COLUMNS = ["customer_id", "name", "industry", "segment", "country"]

_IBM_RENAME = {
    "invoiceNumber": "invoice_id",
    "customerID": "customer_id",
    "countryCode": "country",
    "InvoiceDate": "invoice_date",
    "DueDate": "due_date",
    "SettledDate": "settled_date",
    "InvoiceAmount": "amount",
}

_NAME_A = ["Nord", "Alpen", "Rhein", "Elb", "Isar", "Main", "Spree", "Weser", "Hansa", "Baltic",
           "Atlas", "Orion", "Vega", "Delta", "Kappa", "Lumen", "Ferro", "Terra", "Aqua", "Volt"]
_NAME_B = ["wind", "stahl", "werk", "licht", "tech", "form", "bau", "logik", "trans", "chem",
           "med", "net", "pack", "flow", "tec", "mobil", "data", "plast", "food", "print"]
_NAME_C = ["GmbH", "AG", "SE", "KG", "Ltd", "B.V.", "S.A.", "Oy", "AB", "S.r.l."]
INDUSTRIES = ["Manufacturing", "Wholesale", "Logistics", "Retail", "Construction",
              "Healthcare", "Software", "Energy", "Food & Beverage", "Automotive"]
SEGMENTS = ["SMB", "Mid-market", "Enterprise"]


def company_name(key: str) -> str:
    """Deterministic, obviously fictional company name for a customer id."""
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return f"{_NAME_A[h % 20]}{_NAME_B[(h // 20) % 20]} {_NAME_C[(h // 400) % 10]}"


def validate_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on data that would silently corrupt training."""
    missing = set(INVOICE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df["invoice_id"].duplicated().any():
        raise ValueError("duplicate invoice ids")
    if (df["due_date"] < df["invoice_date"]).any():
        raise ValueError("due date before invoice date")
    if (df["settled_date"] < df["invoice_date"]).any():
        raise ValueError("settled before invoiced")
    if (df["amount"] <= 0).any():
        raise ValueError("non-positive invoice amount")
    expected_late = (df["settled_date"] - df["due_date"]).dt.days.clip(lower=0)
    if not (expected_late == df["days_late"]).all():
        raise ValueError("days_late is inconsistent with due/settled dates")
    return df


def load_ibm(path: str | None = None, amount_scale: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    raw = pd.read_csv(resolve(path or cfg["data"]["ibm_path"]), dtype={"invoiceNumber": str, "customerID": str})
    df = raw.rename(columns=_IBM_RENAME)
    for c in ["invoice_date", "due_date", "settled_date"]:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y")
    df["country"] = df["country"].astype(str)
    df["amount"] = df["amount"].astype(float) * amount_scale
    df["disputed"] = raw["Disputed"].eq("Yes")
    df["is_paperless"] = raw["PaperlessBill"].eq("Electronic")
    df["days_to_settle"] = (df["settled_date"] - df["invoice_date"]).dt.days
    df["days_late"] = (df["settled_date"] - df["due_date"]).dt.days.clip(lower=0)
    df = validate_invoices(df[INVOICE_COLUMNS].sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True))

    # The IBM sample has no customer master data: add a small synthetic one for the UI.
    cust = df.groupby("customer_id", as_index=False)["country"].first()
    rng = np.random.default_rng(7)
    cust["name"] = cust["customer_id"].map(company_name)
    cust["industry"] = rng.choice(INDUSTRIES, len(cust))
    cust["segment"] = rng.choice(SEGMENTS, len(cust), p=[0.5, 0.35, 0.15])
    return df, cust[CUSTOMER_COLUMNS]


def load_synthetic(out_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    d = resolve(out_dir or cfg["data"]["synthetic"]["out_dir"])
    inv_path, cust_path = d / "invoices.csv", d / "customers.csv"
    if not inv_path.exists():
        from .synthetic import generate
        generate(save=True)
    inv = pd.read_csv(inv_path, dtype={"invoice_id": str, "customer_id": str, "country": str},
                      parse_dates=["invoice_date", "due_date", "settled_date"])
    cust = pd.read_csv(cust_path, dtype={"customer_id": str, "country": str})
    return validate_invoices(inv), cust


def load_data(source: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    source = source or cfg["data"]["source"]
    scale = cfg["data"]["amount_scale"].get(source, 1)
    if source == "ibm":
        return load_ibm(amount_scale=scale)
    if source == "synthetic":
        return load_synthetic()
    raise ValueError(f"unknown data source {source!r}")
