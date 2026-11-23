"""SHAP explanations turned into sentences a collector can act on."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .features import CATEGORICAL_FEATURES


def _fmt_money(x: float) -> str:
    return f"€{x:,.0f}"


def _sentence(feature: str, row: pd.Series) -> str:
    v = row.get(feature)
    n = int(row.get("hist_n_settled", 0) or 0)
    t = {
        "hist_late_rate": lambda: f"Paid {v:.0%} of past invoices late ({n} settled)",
        "hist_mean_days_late": lambda: f"Average delay on past invoices: {v:.1f} days",
        "hist_max_days_late": lambda: f"Worst past delay: {v:.0f} days",
        "hist_recent_mean_days_late": lambda: f"Average delay on the last 3 invoices: {v:.1f} days",
        "hist_trend_days_late": lambda: (f"Recent delays are {v:+.1f} days vs. the customer's long-run average"),
        "hist_months_worsening": lambda: (f"Payment delays have grown {int(v)} month{'s' if v != 1 else ''} in a row"
                                          if v else "No recent month-on-month worsening"),
        "hist_ewm_days_late": lambda: f"Recency-weighted delay: {v:.1f} days",
        "hist_dispute_rate": lambda: f"Disputed {v:.0%} of past invoices",
        "hist_settle_ratio": lambda: f"Pays after {v:.0%} of the payment term on average",
        "days_since_last_late": lambda: ("Has never paid late" if v >= 9999 else f"Last late payment {v:.0f} days ago"),
        "open_n": lambda: f"{int(v)} other invoices currently open",
        "open_amount_log": lambda: f"{_fmt_money(np.expm1(v))} currently open with this customer",
        "open_overdue_n": lambda: f"{int(v)} invoices already overdue",
        "amount_vs_customer_mean": lambda: f"Invoice is {v:.1f}x this customer's usual amount",
        "customer_tenure_days": lambda: f"Customer for {v / 365:.1f} years",
        "is_new_customer": lambda: "New customer, no payment history" if v else "Has payment history",
        "hist_n_settled": lambda: f"{int(v)} past invoices settled",
        "log_amount": lambda: f"Invoice amount {_fmt_money(np.expm1(v))}",
        "terms_days": lambda: f"Payment terms: {int(v)} days",
        "is_paperless": lambda: "Sent electronically" if v else "Sent on paper",
        "is_december": lambda: "Issued in December (year-end slowdown)" if v else "Not issued in December",
        "hist_recent_late_rate": lambda: f"Paid {v:.0%} of the last 10 invoices late",
        "hist_late_rate_90d": lambda: f"Late-payment rate over the last 90 days: {v:.0%}",
        "is_quarter_end": lambda: "Issued in a quarter-end month" if v else "Not a quarter-end month",
        "country": lambda: f"Customer country: {v}",
    }.get(feature)
    return t() if t else f"{feature} = {v}"


class Explainer:
    def __init__(self, model: Pipeline):
        self.pre = model.named_steps["pre"]
        self.booster = model.named_steps["clf"]
        self.explainer = shap.TreeExplainer(self.booster)
        self.names = list(self.pre.get_feature_names_out())

    def shap_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds) per original feature; one-hot columns summed back."""
        sv = self.explainer.shap_values(self.pre.transform(X))
        df = pd.DataFrame(sv, columns=self.names, index=X.index)
        for cat in CATEGORICAL_FEATURES:
            cols = [c for c in self.names if c.startswith(f"{cat}_")]
            df[cat] = df[cols].sum(axis=1)
            df = df.drop(columns=cols)
        return df

    def reasons(self, X: pd.DataFrame, top_k: int = 4) -> list[list[dict]]:
        sv = self.shap_frame(X)
        out = []
        for idx in X.index:
            s = sv.loc[idx].sort_values(key=np.abs, ascending=False).head(top_k)
            out.append([{"feature": f, "impact": round(float(v), 4),
                         "direction": "raises risk" if v > 0 else "lowers risk",
                         "text": _sentence(f, X.loc[idx])} for f, v in s.items()])
        return out

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        return self.shap_frame(X).abs().mean().sort_values(ascending=False)
