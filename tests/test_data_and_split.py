import pandas as pd
import pytest

from cashflow_radar.data.load import INVOICE_COLUMNS, load_data, validate_invoices
from cashflow_radar.data.synthetic import generate
from cashflow_radar.split import expanding_window_folds, time_split


def test_ibm_sample_loads_and_validates():
    inv, cust = load_data("ibm")
    assert len(inv) == 2466
    assert inv["customer_id"].nunique() == 100 == len(cust)
    assert list(inv.columns) == INVOICE_COLUMNS
    assert 0.3 < (inv["days_late"] > 0).mean() < 0.4


def test_synthetic_is_deterministic_and_valid():
    a, _ = generate(n_customers=15, seed=1)
    b, _ = generate(n_customers=15, seed=1)
    pd.testing.assert_frame_equal(a, b)
    validate_invoices(a)


def test_validation_rejects_inconsistent_days_late():
    inv, _ = generate(n_customers=5, seed=2)
    inv.loc[0, "days_late"] += 3
    with pytest.raises(ValueError, match="days_late"):
        validate_invoices(inv)


def test_time_split_is_chronological_and_purges_unknown_labels():
    inv, _ = load_data("ibm")
    s = time_split(inv, 0.6, 0.2)
    assert s.train["invoice_date"].max() < s.valid_start <= s.valid["invoice_date"].min()
    assert s.valid["invoice_date"].max() < s.test_start <= s.test["invoice_date"].min()
    # every training label was observable at the validation cut-off
    assert (s.train["settled_date"] < s.valid_start).all()
    assert (s.valid["settled_date"] < s.test_start).all()
    assert s.n_purged_train > 0


def test_expanding_window_folds_never_look_ahead():
    inv, _ = load_data("ibm")
    df = inv.set_index("invoice_id", drop=False)
    folds = list(expanding_window_folds(df, 4))
    assert len(folds) == 4
    for tr, va in folds:
        assert df.loc[tr, "settled_date"].max() < df.loc[va, "invoice_date"].min()
