from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Reason(BaseModel):
    feature: str
    impact: float = Field(description="SHAP value in log-odds; positive raises late-payment risk")
    direction: Literal["raises risk", "lowers risk"]
    text: str


class WorklistItem(BaseModel):
    priority_rank: int
    invoice_id: str
    customer_id: str
    customer_name: str
    country: str
    segment: str | None = None
    amount: float
    invoice_date: date
    due_date: date
    status: Literal["open", "overdue"]
    days_to_due: int
    p_late: float
    expected_days_late: float
    expected_net_value: float
    breakeven_p: float
    action: Literal["touch", "monitor"]
    reasons: list[Reason]


class Worklist(BaseModel):
    as_of: date
    total: int
    items: list[WorklistItem]


class PolicyResult(BaseModel):
    policy: str
    touches: int
    dso_days: float
    avg_overdue_balance: float
    net_savings: float


class Summary(BaseModel):
    as_of: date
    model_version: str
    source: str
    open_invoices: int
    open_balance: float
    overdue_balance: float
    expected_late_balance: float
    recommended_touches: int
    recommended_touch_value: float
    test_metrics: dict[str, dict[str, float]]
    policy_simulation: list[PolicyResult]
    cost_model: dict[str, float]
    capacity_share: float


class HistoryPoint(BaseModel):
    invoice_id: str
    invoice_date: date
    settled_date: date
    amount: float
    days_late: int


class InvoiceDetail(WorklistItem):
    industry: str | None = None
    history: list[HistoryPoint]


class ScoreRequest(BaseModel):
    customer_id: str
    amount: float = Field(gt=0)
    invoice_date: date
    due_date: date
    is_paperless: bool = True
    country: str | None = Field(default=None, description="defaults to the customer's country")

    @model_validator(mode="after")
    def _dates(self):
        if self.due_date < self.invoice_date:
            raise ValueError("due_date must not be before invoice_date")
        return self


class ScoreResponse(BaseModel):
    customer_id: str
    p_late: float
    expected_days_late: float
    expected_days_late_if_late: float
    expected_net_value: float
    breakeven_p: float
    action: Literal["touch", "monitor"]
    reasons: list[Reason]
    model_version: str
