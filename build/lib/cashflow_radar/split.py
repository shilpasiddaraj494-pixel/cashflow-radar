"""Chronological train / validation / test split with label purging.

Random K-fold on invoices leaks in two ways: (1) the model trains on invoices issued after
the ones it is tested on, and (2) it trains on labels that did not exist yet at the time.
Here every boundary is a date, and an invoice only counts as a training example for a
period if it was *settled* before that period ended (its label was observable then).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    valid_start: pd.Timestamp
    test_start: pd.Timestamp
    n_purged_train: int
    n_purged_valid: int

    def summary(self) -> dict:
        return {
            "train_rows": len(self.train), "valid_rows": len(self.valid), "test_rows": len(self.test),
            "valid_start": str(self.valid_start.date()), "test_start": str(self.test_start.date()),
            "purged_train": self.n_purged_train, "purged_valid": self.n_purged_valid,
        }


def time_split(inv: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2) -> TimeSplit:
    inv = inv.sort_values("invoice_date").reset_index(drop=True)
    dates = inv["invoice_date"]
    valid_start = dates.quantile(train_frac).normalize()
    test_start = dates.quantile(train_frac + valid_frac).normalize()

    train_all = inv[dates < valid_start]
    valid_all = inv[(dates >= valid_start) & (dates < test_start)]
    train = train_all[train_all["settled_date"] < valid_start]
    valid = valid_all[valid_all["settled_date"] < test_start]
    test = inv[dates >= test_start]
    return TimeSplit(train, valid, test, valid_start, test_start,
                     len(train_all) - len(train), len(valid_all) - len(valid))


def expanding_window_folds(df: pd.DataFrame, n_folds: int = 4, min_train_frac: float = 0.4):
    """Yield (train_idx, val_idx) for expanding-window CV inside one period, with purging."""
    df = df.sort_values("invoice_date")
    dates = df["invoice_date"]
    cuts = [dates.quantile(min_train_frac + (1 - min_train_frac) * k / n_folds) for k in range(n_folds + 1)]
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
        tr = df[(dates < lo) & (df["settled_date"] < lo)].index
        va = df[(dates >= lo) & (dates < hi)].index
        if len(tr) and len(va):
            yield tr, va
