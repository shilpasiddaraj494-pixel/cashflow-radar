import numpy as np
import pandas as pd

from cashflow_radar.decision import (
    breakeven_probability,
    expected_net_value,
    global_threshold_curve,
    simulate_policies,
    touch_benefit,
)
from cashflow_radar.modeling import Calibrator, expected_calibration_error

CM = {"cost_per_touch": 25.0, "cost_of_capital": 0.10, "bad_debt_rate_if_late": 0.01, "touch_effectiveness": 0.35}


def test_touch_benefit_is_zero_for_on_time_invoices():
    b = touch_benefit(np.array([10_000.0, 10_000.0]), np.array([0, 20]), CM)
    assert b[0] == 0
    assert np.isclose(b[1], 0.35 * 10_000 * (0.10 / 365 * 20 + 0.01))


def test_breakeven_rule_is_consistent():
    amount, days = np.array([50_000.0, 2_000.0]), np.array([15.0, 15.0])
    p_star = breakeven_probability(amount, days, CM)
    assert p_star[0] < p_star[1]  # large invoices are worth a touch at lower risk
    assert np.allclose(expected_net_value(p_star, amount, days, CM)[p_star < 1], 0)
    assert p_star[1] == 1.0  # a €2k invoice never pays for a €25 touch


def test_threshold_curve_extremes():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    amount = np.full(500, 5_000.0)
    dl = np.where(rng.random(500) < p, 10, 0)
    curve = global_threshold_curve(p, amount, dl, CM)
    assert curve["touches"].is_monotonic_decreasing


def test_simulation_oracle_dominates_and_no_touch_is_baseline():
    rng = np.random.default_rng(1)
    n = 400
    test = pd.DataFrame({
        "invoice_id": [f"I{i}" for i in range(n)], "customer_id": [f"C{i % 20}" for i in range(n)],
        "invoice_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D"),
        "amount": rng.lognormal(9, 1, n), "days_late": rng.integers(0, 30, n) * (rng.random(n) < 0.4),
    })
    test["days_to_settle"] = 30 + test["days_late"]
    scores = pd.DataFrame({"invoice_id": test["invoice_id"], "expected_net_value": rng.normal(0, 50, n)})
    names = pd.Series({f"C{i}": f"Name {i}" for i in range(20)})
    sim = simulate_policies(test, scores, names, CM, 0.15).set_index("policy")
    assert sim.loc["no proactive touches", "net_savings"] == 0
    assert sim["avg_overdue_balance"].idxmin() == "oracle (perfect foresight)"
    assert sim.loc["oracle (perfect foresight)", "dso_days"] <= sim.loc["no proactive touches", "dso_days"]


def test_calibrator_fixes_a_miscalibrated_model():
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.05, 0.95, 6000)
    y = (rng.random(6000) < true_p).astype(int)
    raw = true_p ** 2.5  # systematically under-confident about lateness
    cal = Calibrator().fit(raw, y)
    assert cal.method != "none"
    assert expected_calibration_error(y, cal.transform(raw)) < expected_calibration_error(y, raw) / 3
