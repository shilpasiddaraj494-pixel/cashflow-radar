export type Reason = {
  feature: string;
  impact: number;
  direction: "raises risk" | "lowers risk";
  text: string;
};

export type WorklistItem = {
  priority_rank: number;
  invoice_id: string;
  customer_id: string;
  customer_name: string;
  country: string;
  segment: string | null;
  amount: number;
  invoice_date: string;
  due_date: string;
  status: "open" | "overdue";
  days_to_due: number;
  p_late: number;
  expected_days_late: number;
  expected_net_value: number;
  breakeven_p: number;
  action: "touch" | "monitor";
  reasons: Reason[];
  shap_base?: number | null;
  shap_score?: number | null;
};

export type Worklist = { as_of: string; total: number; items: WorklistItem[] };

export type HistoryPoint = {
  invoice_id: string;
  invoice_date: string;
  settled_date: string;
  amount: number;
  days_late: number;
};

export type InvoiceDetail = WorklistItem & { industry: string | null; history: HistoryPoint[] };

export type PolicyResult = {
  policy: string;
  touches: number;
  dso_days: number;
  avg_overdue_balance: number;
  net_savings: number;
};

export type Metrics = { roc_auc: number; pr_auc: number; brier: number; log_loss: number; ece: number };

export type Summary = {
  as_of: string;
  model_version: string;
  source: string;
  open_invoices: number;
  open_balance: number;
  overdue_balance: number;
  expected_late_balance: number;
  recommended_touches: number;
  recommended_touch_value: number;
  test_metrics: Record<string, Metrics>;
  policy_simulation: PolicyResult[];
  cost_model: Record<string, number>;
  capacity_share: number;
};

export type Filters = {
  action: "all" | "touch" | "monitor";
  status: "all" | "open" | "overdue";
  q: string;
  offset: number;
  limit: number;
  sort: "priority" | "amount" | "p_late" | "due_date" | "customer";
  desc: boolean;
};

/* ---- Static snapshot mode: data embedded in the page, no server needed. ---- */
type CompactItem = Omit<WorklistItem, "reasons" | "customer_name" | "country" | "segment"> & {
  reasons: [string, number, string][];
};
type CustomerInfo = { customer_name: string; industry: string | null; segment: string | null; country: string };
type StaticData = {
  common: Omit<Summary, "as_of" | "open_invoices" | "open_balance" | "overdue_balance" | "expected_late_balance" | "recommended_touches" | "recommended_touch_value">;
  weeks: { summary: Pick<Summary, "as_of" | "open_invoices" | "open_balance" | "overdue_balance" | "expected_late_balance" | "recommended_touches" | "recommended_touch_value">; items: CompactItem[] }[];
  histories: Record<string, HistoryPoint[]>;
  details: Record<string, CustomerInfo>;
};
const STATIC = (window as unknown as { __CFR_DATA__?: StaticData }).__CFR_DATA__;

const expanded = new Map<string, WorklistItem[]>();
function weekItems(asOf: string): WorklistItem[] {
  if (!expanded.has(asOf)) {
    const w = STATIC!.weeks.find((x) => x.summary.as_of === asOf) ?? STATIC!.weeks[STATIC!.weeks.length - 1];
    expanded.set(
      asOf,
      w.items.map((i) => {
        const c = STATIC!.details[i.customer_id];
        return {
          ...i,
          customer_name: c?.customer_name ?? i.customer_id,
          country: c?.country ?? "",
          segment: c?.segment ?? null,
          reasons: i.reasons.map(([feature, impact, text]) => ({
            feature,
            impact,
            text,
            direction: impact > 0 ? ("raises risk" as const) : ("lowers risk" as const),
          })),
        };
      }),
    );
  }
  return expanded.get(asOf)!;
}

function staticWorklist(f: Filters, asOf: string): Worklist {
  const q = f.q.trim().toLowerCase();
  const rows = weekItems(asOf).filter(
    (i) =>
      (f.action === "all" || i.action === f.action) &&
      (f.status === "all" || i.status === f.status) &&
      (!q || i.customer_name.toLowerCase().includes(q) || i.invoice_id.toLowerCase().includes(q)),
  );
  const key: Record<Filters["sort"], (i: WorklistItem) => number | string> = {
    priority: (i) => i.priority_rank,
    amount: (i) => i.amount,
    p_late: (i) => i.p_late,
    due_date: (i) => i.due_date,
    customer: (i) => i.customer_name,
  };
  const k = key[f.sort];
  rows.sort((a, b) => {
    const x = k(a),
      y = k(b);
    const c = x < y ? -1 : x > y ? 1 : a.priority_rank - b.priority_rank;
    return f.desc ? -c : c;
  });
  return { as_of: asOf, total: rows.length, items: rows.slice(f.offset, f.offset + f.limit) };
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = STATIC
  ? {
      weeks: async () => STATIC.weeks.map((w) => w.summary.as_of),
      summary: async (asOf?: string): Promise<Summary> => {
        const w = STATIC.weeks.find((x) => x.summary.as_of === asOf) ?? STATIC.weeks[STATIC.weeks.length - 1];
        return { ...STATIC.common, ...w.summary };
      },
      worklist: async (f: Filters, asOf?: string) =>
        staticWorklist(f, asOf ?? STATIC.weeks[STATIC.weeks.length - 1].summary.as_of),
      invoice: async (id: string, asOf?: string): Promise<InvoiceDetail> => {
        const week = asOf ?? STATIC.weeks[STATIC.weeks.length - 1].summary.as_of;
        const it = weekItems(week).find((i) => i.invoice_id === id);
        if (!it) throw new Error(`invoice ${id} not found`);
        const history = (STATIC.histories[it.customer_id] ?? []).filter((h) => h.settled_date < week).slice(-24);
        return { ...it, industry: STATIC.details[it.customer_id]?.industry ?? null, history };
      },
    }
  : {
      weeks: async () => [(await get<Summary>("/api/summary")).as_of],
      summary: (_asOf?: string) => get<Summary>("/api/summary"),
      worklist: (f: Filters, _asOf?: string) => {
        const p = new URLSearchParams({
          action: f.action,
          status: f.status,
          limit: String(f.limit),
          offset: String(f.offset),
          sort: f.sort,
          desc: String(f.desc),
        });
        if (f.q.trim()) p.set("q", f.q.trim());
        return get<Worklist>(`/api/worklist?${p}`);
      },
      invoice: (id: string, _asOf?: string) => get<InvoiceDetail>(`/api/invoices/${encodeURIComponent(id)}`),
    };
