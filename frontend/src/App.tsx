import { useEffect, useState } from "react";
import { api, type Filters, type InvoiceDetail, type Summary, type Worklist, type WorklistItem } from "./api";
import { HistoryChart, PolicyChart, ScoreDrivers } from "./components/Charts";
import { RiskMeter } from "./components/RiskMeter";
import { date, dueLabel, money, moneyCompact, pct } from "./format";
import { useContacted } from "./useContacted";

const PAGE = 25;
type Tab = "worklist" | "impact";

export default function App() {
  const [tab, setTab] = useState<Tab>("worklist");
  const [weeks, setWeeks] = useState<string[]>([]);
  const [week, setWeek] = useState<string | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .weeks()
      .then((w) => {
        setWeeks(w);
        setWeek(w[w.length - 1]);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (week) api.summary(week).then(setSummary).catch((e) => setError(e.message));
  }, [week]);

  return (
    <div className="page">
      <header className="hero">
        <div>
          <h1>CashFlow Radar</h1>
          <p className="tagline">Collections worklist with explainable late-payment risk</p>
        </div>
        {summary && (
          <div className="badge">
            {summary.source === "ibm" ? "IBM sample data" : "Synthetic data"}
            <span>Model {summary.model_version}</span>
          </div>
        )}
      </header>

      <nav className="tabs" aria-label="Sections">
        <button className={tab === "worklist" ? "tab on" : "tab"} onClick={() => setTab("worklist")}>
          Worklist
        </button>
        <button className={tab === "impact" ? "tab on" : "tab"} onClick={() => setTab("impact")}>
          Model evaluation
        </button>
      </nav>

      {error ? (
        <div className="card error-state">
          <h2>The worklist could not load</h2>
          <p>{error}</p>
          <p>
            Check that the API is running on port 8000 and that a worklist has been scored with{" "}
            <code>python -m cashflow_radar.score</code>.
          </p>
        </div>
      ) : !summary || !week ? null : tab === "worklist" ? (
        <WorklistView summary={summary} week={week} weeks={weeks} onWeek={setWeek} onError={setError} />
      ) : (
        <ImpactView s={summary} />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ worklist */

function WorklistView({
  summary,
  week,
  weeks,
  onWeek,
  onError,
}: {
  summary: Summary;
  week: string;
  weeks: string[];
  onWeek: (w: string) => void;
  onError: (m: string) => void;
}) {
  const [list, setList] = useState<Worklist | null>(null);
  const [filters, setFilters] = useState<Filters>({
    action: "touch",
    status: "all",
    q: "",
    offset: 0,
    limit: PAGE,
    sort: "priority",
    desc: false,
  });
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<InvoiceDetail | null>(null);
  const contacted = useContacted();

  useEffect(() => {
    const t = setTimeout(() => {
      api
        .worklist(filters, week)
        .then((w) => {
          setList(w);
          setSelected((cur) => (w.items.some((i) => i.invoice_id === cur) ? cur : (w.items[0]?.invoice_id ?? null)));
        })
        .catch((e) => onError(e.message));
    }, 100);
    return () => clearTimeout(t);
  }, [filters, week, onError]);

  // Only load the detail once the list for the current week contains the selection,
  // so switching weeks never asks for an invoice that week doesn't have.
  useEffect(() => {
    if (!selected || !list || list.as_of !== week || !list.items.some((i) => i.invoice_id === selected)) return;
    let live = true;
    api
      .invoice(selected, week)
      .then((d) => live && setDetail(d))
      .catch((e) => live && onError(e.message));
    return () => {
      live = false;
    };
  }, [selected, list, week, onError]);

  const set = (patch: Partial<Filters>) => setFilters((f) => ({ ...f, offset: 0, ...patch }));
  const sortBy = (key: Filters["sort"]) =>
    setFilters((f) => ({
      ...f,
      offset: 0,
      sort: key,
      desc: f.sort === key ? !f.desc : key === "amount" || key === "p_late",
    }));

  return (
    <>
      <section className="kpis card" aria-label={`Portfolio on ${date(week)}`}>
        <Kpi label="Open receivables" value={moneyCompact(summary.open_balance)} note={`${summary.open_invoices.toLocaleString()} invoices`} />
        <Kpi
          label="Already overdue"
          value={moneyCompact(summary.overdue_balance)}
          note={`${pct(summary.overdue_balance / summary.open_balance)} of open balance`}
        />
        <Kpi label="Expected to pay late" value={moneyCompact(summary.expected_late_balance)} note="risk-weighted balance" />
        <Kpi
          label="Worth a reminder this week"
          value={summary.recommended_touches.toLocaleString()}
          note={`${money(summary.recommended_touch_value)} expected net value`}
          accent
        />
      </section>

      <div className="columns">
        <section className="card list-card">
          <div className="card-head">
            <div>
              <h2>Invoice priority</h2>
              <p className="muted">Ranked by the expected value of a reminder: risk × balance × days late</p>
            </div>
            <label className="week">
              <span>Scoring week</span>
              <select value={week} onChange={(e) => onWeek(e.target.value)} disabled={weeks.length < 2}>
                {weeks.map((w) => (
                  <option key={w} value={w}>
                    {date(w)}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="filters" role="toolbar" aria-label="Filter the worklist">
            <Choice
              label="Show"
              value={filters.action}
              options={[
                ["touch", "Worth a reminder"],
                ["monitor", "Monitor only"],
                ["all", "All open"],
              ]}
              onChange={(v) => set({ action: v as Filters["action"] })}
            />
            <Choice
              label="Status"
              value={filters.status}
              options={[
                ["all", "Any"],
                ["open", "Not yet due"],
                ["overdue", "Overdue"],
              ]}
              onChange={(v) => set({ status: v as Filters["status"] })}
            />
            <input
              className="search"
              type="search"
              aria-label="Search customer or invoice"
              placeholder="Search customer or invoice"
              value={filters.q}
              onChange={(e) => set({ q: e.target.value })}
            />
          </div>

          {list && list.items.length === 0 ? (
            <p className="empty">No invoices match these filters. Widen the status filter or clear the search.</p>
          ) : (
            <table className="inv-table">
              <thead>
                <tr>
                  <SortTh label="Invoice" k="priority" f={filters} onSort={sortBy} />
                  <SortTh label="Due date" k="due_date" f={filters} onSort={sortBy} />
                  <SortTh label="Balance" k="amount" f={filters} onSort={sortBy} className="num" />
                  <SortTh label="Late risk" k="p_late" f={filters} onSort={sortBy} />
                  <th scope="col">Next step</th>
                </tr>
              </thead>
              <tbody>
                {list?.items.map((it) => (
                  <Row
                    key={it.invoice_id}
                    it={it}
                    active={it.invoice_id === selected}
                    done={contacted.has(it.invoice_id)}
                    onSelect={setSelected}
                  />
                ))}
              </tbody>
            </table>
          )}

          {list && list.total > PAGE && (
            <nav className="pager" aria-label="Pages">
              <span className="muted">
                {filters.offset + 1}–{Math.min(filters.offset + PAGE, list.total)} of {list.total.toLocaleString()} invoices
              </span>
              <button className="btn" disabled={filters.offset === 0} onClick={() => setFilters((f) => ({ ...f, offset: f.offset - PAGE }))}>
                Previous
              </button>
              <button
                className="btn"
                disabled={filters.offset + PAGE >= list.total}
                onClick={() => setFilters((f) => ({ ...f, offset: f.offset + PAGE }))}
              >
                Next
              </button>
            </nav>
          )}
        </section>

        <aside className="card detail-card" aria-live="polite">
          {detail ? (
            <Detail
              d={detail}
              costPerTouch={summary.cost_model.cost_per_touch}
              done={contacted.has(detail.invoice_id)}
              onToggleDone={() => contacted.toggle(detail.invoice_id)}
            />
          ) : (
            <p className="empty">Select an invoice to see why it is ranked where it is.</p>
          )}
        </aside>
      </div>
    </>
  );
}

function Kpi({ label, value, note, accent }: { label: string; value: string; note: string; accent?: boolean }) {
  return (
    <div className={accent ? "kpi kpi-accent" : "kpi"}>
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">{value}</span>
      <span className="kpi-note">{note}</span>
    </div>
  );
}

function Choice({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: [string, string][];
  onChange: (v: string) => void;
}) {
  return (
    <div className="choice" role="radiogroup" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={v} role="radio" aria-checked={value === v} className={value === v ? "on" : ""} onClick={() => onChange(v)}>
          {text}
        </button>
      ))}
    </div>
  );
}

function SortTh({
  label,
  k,
  f,
  onSort,
  className,
}: {
  label: string;
  k: Filters["sort"];
  f: Filters;
  onSort: (k: Filters["sort"]) => void;
  className?: string;
}) {
  const active = f.sort === k;
  return (
    <th scope="col" className={className} aria-sort={active ? (f.desc ? "descending" : "ascending") : "none"}>
      <button className={active ? "sort on" : "sort"} onClick={() => onSort(k)}>
        {label}
        <span className="sort-arrow" aria-hidden="true">
          {active ? (f.desc ? " ↓" : " ↑") : ""}
        </span>
      </button>
    </th>
  );
}

function Row({
  it,
  active,
  done,
  onSelect,
}: {
  it: WorklistItem;
  active: boolean;
  done: boolean;
  onSelect: (id: string) => void;
}) {
  const overdue = it.status === "overdue";
  const next = done ? "Contacted" : it.action === "touch" ? (overdue ? "Chase" : "Remind") : "Wait";
  return (
    <tr
      className={[active && "active", done && "done"].filter(Boolean).join(" ")}
      tabIndex={0}
      onClick={() => onSelect(it.invoice_id)}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onSelect(it.invoice_id))}
    >
      <td>
        <div className="inv-id">{it.invoice_id}</div>
        <div className="sub">{it.customer_name}</div>
      </td>
      <td>
        <div>{date(it.due_date)}</div>
        <div className={overdue ? "sub overdue" : "sub"}>{dueLabel(it.days_to_due)}</div>
      </td>
      <td className="num">{money(it.amount)}</td>
      <td>
        <div className="risk-cell">
          <span className="risk-pct">{overdue ? "Late" : pct(it.p_late, 1)}</span>
          <RiskMeter p={it.p_late} breakeven={it.breakeven_p} overdue={overdue} width={90} />
        </div>
      </td>
      <td>
        <span className={`step step-${next.toLowerCase()}`}>{next}</span>
      </td>
    </tr>
  );
}

function Detail({
  d,
  costPerTouch,
  done,
  onToggleDone,
}: {
  d: InvoiceDetail;
  costPerTouch: number;
  done: boolean;
  onToggleDone: () => void;
}) {
  const overdue = d.status === "overdue";
  const daysIfLate = overdue ? d.expected_days_late : d.expected_days_late / Math.max(d.p_late, 1e-6);
  return (
    <article className="detail">
      <header className="detail-head">
        <div>
          <h2>{d.invoice_id}</h2>
          <p className="muted">
            {d.customer_name}, {money(d.amount)} outstanding
          </p>
          <p className="muted small">
            {[d.segment, d.industry, d.country].filter(Boolean).join(", ")}. Issued {date(d.invoice_date)}, due{" "}
            {date(d.due_date)}
          </p>
        </div>
        <div className="big-risk">
          <span className={overdue ? "big-risk-value overdue" : "big-risk-value"}>{overdue ? `${-d.days_to_due}d` : pct(d.p_late, 1)}</span>
          <span className="big-risk-label">{overdue ? "Overdue" : "Late-payment risk"}</span>
        </div>
      </header>

      <div className={d.action === "touch" ? "verdict verdict-touch" : "verdict"}>
        <div>
          <strong>
            {d.action === "touch"
              ? overdue
                ? "Chase this invoice now"
                : "Send a reminder before the due date"
              : "No action yet"}
          </strong>
          <p>
            {d.action === "touch"
              ? `Expected value of a ${money(costPerTouch)} reminder: ${money(d.expected_net_value)} after its cost.`
              : `A ${money(costPerTouch)} reminder would cost more than it is likely to save.`}
          </p>
        </div>
        <button className={done ? "btn btn-done" : "btn btn-primary"} onClick={onToggleDone}>
          {done ? "Contacted ✓" : "Mark contacted"}
        </button>
      </div>

      <section className="block">
        <h3>What drove this score</h3>
        <ScoreDrivers reasons={d.reasons} base={d.shap_base ?? null} score={d.shap_score ?? null} />
      </section>

      <section className="block">
        <h3>Payment history</h3>
        <HistoryChart history={d.history} />
      </section>

      <section className="block">
        <h3>Decision</h3>
        <dl className="math">
          <div>
            <dt>Chance of paying late</dt>
            <dd>{overdue ? "already late" : pct(d.p_late, 1)}</dd>
          </div>
          <div>
            <dt>Expected days late, if late</dt>
            <dd>{daysIfLate.toFixed(1)}</dd>
          </div>
          <div>
            <dt>Risk at which a {money(costPerTouch)} reminder breaks even</dt>
            <dd>{pct(d.breakeven_p, 1)}</dd>
          </div>
          <div className="math-total">
            <dt>Expected net value of a reminder</dt>
            <dd>{money(d.expected_net_value)}</dd>
          </div>
        </dl>
      </section>
    </article>
  );
}

/* ------------------------------------------------------------------ evaluation */

function ImpactView({ s }: { s: Summary }) {
  const rows: [string, string][] = [
    ["heuristic_customer_late_rate", "Customer's past late rate"],
    ["logistic_regression", "Logistic regression"],
    ["xgboost_uncalibrated", "XGBoost"],
    ["xgboost_calibrated", "XGBoost, calibrated (in use)"],
  ];
  const model = s.policy_simulation.find((p) => p.policy.startsWith("CashFlow"));
  const alpha = s.policy_simulation.find((p) => p.policy.startsWith("alphabetical"));
  return (
    <div className="columns columns-even">
      <section className="card">
        <h2>Does ranking by risk pay off?</h2>
        <p className="muted prose">
          A replay of the held-out test period. Each week collectors can contact {pct(s.capacity_share)} of new invoices,
          and a contact is assumed to remove {pct(s.cost_model.touch_effectiveness)} of an invoice's lateness.
          {model && alpha && (
            <>
              {" "}
              Ranking by the model brings the average overdue balance to {moneyCompact(model.avg_overdue_balance)}, against{" "}
              {moneyCompact(alpha.avg_overdue_balance)} when the list is worked alphabetically, and DSO from{" "}
              {alpha.dso_days.toFixed(1)} to {model.dso_days.toFixed(1)} days.
            </>
          )}
        </p>
        <PolicyChart rows={s.policy_simulation} />
        <table className="plain-table">
          <thead>
            <tr>
              <th scope="col">Policy</th>
              <th scope="col" className="num">
                Contacts
              </th>
              <th scope="col" className="num">
                DSO (days)
              </th>
              <th scope="col" className="num">
                Avg. overdue
              </th>
              <th scope="col" className="num">
                Net savings
              </th>
            </tr>
          </thead>
          <tbody>
            {s.policy_simulation.map((p) => (
              <tr key={p.policy} className={p.policy.startsWith("CashFlow") ? "in-use" : ""}>
                <th scope="row">{p.policy}</th>
                <td className="num">{p.touches.toLocaleString()}</td>
                <td className="num">{p.dso_days.toFixed(2)}</td>
                <td className="num">{money(p.avg_overdue_balance)}</td>
                <td className="num">{money(p.net_savings)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>How good is the model?</h2>
        <p className="muted prose">
          Scored on months the model never saw. Higher AUC is better; lower Brier score and calibration error (ECE) are
          better.
        </p>
        <table className="plain-table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col" className="num">
                ROC-AUC
              </th>
              <th scope="col" className="num">
                PR-AUC
              </th>
              <th scope="col" className="num">
                Brier
              </th>
              <th scope="col" className="num">
                ECE
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([k, label]) => {
              const m = s.test_metrics[k];
              return (
                <tr key={k} className={k === "xgboost_calibrated" ? "in-use" : ""}>
                  <th scope="row">{label}</th>
                  <td className="num">{m.roc_auc.toFixed(3)}</td>
                  <td className="num">{m.pr_auc.toFixed(3)}</td>
                  <td className="num">{m.brier.toFixed(3)}</td>
                  <td className="num">{m.ece.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <h3 className="sub-h">Assumptions behind the numbers</h3>
        <dl className="math">
          <div>
            <dt>Cost of one reminder</dt>
            <dd>{money(s.cost_model.cost_per_touch)}</dd>
          </div>
          <div>
            <dt>Cost of capital (annual)</dt>
            <dd>{pct(s.cost_model.cost_of_capital)}</dd>
          </div>
          <div>
            <dt>Write-off rate on late invoices</dt>
            <dd>{pct(s.cost_model.bad_debt_rate_if_late, 1)}</dd>
          </div>
          <div>
            <dt>Lateness removed by a reminder</dt>
            <dd>{pct(s.cost_model.touch_effectiveness)}</dd>
          </div>
        </dl>
      </section>
    </div>
  );
}
