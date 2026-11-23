import type { HistoryPoint, PolicyResult, Reason } from "../api";
import { date, days, money, moneyCompact } from "../format";
import { useTooltip } from "./Tooltip";

/** SHAP reasons as a diverging bar around zero, sentence always visible. */
export function ReasonBars({ reasons }: { reasons: Reason[] }) {
  const tip = useTooltip();
  const model = reasons.filter((r) => r.feature !== "days_overdue");
  const max = Math.max(0.05, ...model.map((r) => Math.abs(r.impact)));
  return (
    <ul className="reasons">
      {reasons.map((r) =>
        r.feature === "days_overdue" ? (
          <li key={r.feature} className="reason reason-overdue">
            <span className="status-pill status-critical">
              <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                <circle cx="6" cy="6" r="5.5" fill="currentColor" />
                <rect x="5.25" y="2.6" width="1.5" height="4.4" rx=".75" fill="var(--surface-1)" />
                <circle cx="6" cy="8.8" r=".9" fill="var(--surface-1)" />
              </svg>
              Overdue
            </span>
            <span>{r.text}</span>
          </li>
        ) : (
          <li
            key={r.feature}
            className="reason"
            onMouseMove={(e) =>
              tip.show(
                e,
                <>
                  <strong>{r.direction === "raises risk" ? "Raises" : "Lowers"} late-payment risk</strong>
                  <div>
                    SHAP {r.impact > 0 ? "+" : ""}
                    {r.impact.toFixed(2)} log-odds
                  </div>
                </>,
              )
            }
            onMouseLeave={tip.hide}
          >
            <span className="reason-bar" aria-hidden="true">
              <span className="reason-axis" />
              <span
                className={r.impact > 0 ? "reason-fill reason-up" : "reason-fill reason-down"}
                style={{ width: `${(Math.abs(r.impact) / max) * 50}%` }}
              />
            </span>
            <span className="reason-text">
              <span className="reason-dir">{r.impact > 0 ? "▲" : "▼"}</span> {r.text}
            </span>
          </li>
        ),
      )}
    </ul>
  );
}

/** Days late on the customer's recent settled invoices (single series, oldest left). */
export function HistoryChart({ history }: { history: HistoryPoint[] }) {
  const tip = useTooltip();
  if (!history.length) return <p className="empty">New customer: no settled invoices yet.</p>;
  const W = 420;
  const H = 132;
  const pad = { l: 30, r: 6, t: 8, b: 20 };
  const max = Math.max(10, ...history.map((h) => h.days_late));
  const niceMax = Math.ceil(max / 10) * 10;
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const step = iw / history.length;
  const bw = Math.max(3, Math.min(14, step - 2));
  const ticks = [0, niceMax / 2, niceMax];
  const yOf = (v: number) => pad.t + ih - (v / niceMax) * ih;
  const late = history.filter((h) => h.days_late > 0).length;
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Days late on the last ${history.length} settled invoices`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={yOf(t)} y2={yOf(t)} className={t === 0 ? "axis-base" : "grid"} />
            <text x={pad.l - 6} y={yOf(t) + 3.5} className="axis-label" textAnchor="end">
              {t}
            </text>
          </g>
        ))}
        {history.map((h, i) => {
          const cx = pad.l + step * i + step / 2;
          const onTime = h.days_late === 0;
          return (
            <g
              key={h.invoice_id}
              onMouseMove={(e) =>
                tip.show(
                  e,
                  <>
                    <strong>{onTime ? "Paid on time" : `${days(h.days_late)} late`}</strong>
                    <div>
                      {h.invoice_id}, {money(h.amount)}
                    </div>
                    <div className="tooltip-muted">
                      Issued {date(h.invoice_date)}, paid {date(h.settled_date)}
                    </div>
                  </>,
                )
              }
              onMouseLeave={tip.hide}
            >
              <rect x={cx - step / 2} y={pad.t} width={step} height={ih} fill="transparent" />
              {onTime ? (
                <circle cx={cx} cy={yOf(0) - 4} r={2.5} className="dot-ontime" />
              ) : (
                <path d={barPath(cx - bw / 2, yOf(h.days_late), bw, yOf(0) - yOf(h.days_late))} className="bar-series" />
              )}
            </g>
          );
        })}
        <text x={pad.l} y={H - 4} className="axis-label">
          older
        </text>
        <text x={W - pad.r} y={H - 4} className="axis-label" textAnchor="end">
          most recent
        </text>
      </svg>
      <figcaption>
        Days late, last {history.length} settled invoices. {late} of {history.length} paid late.
      </figcaption>
    </figure>
  );
}

/** Bar with 4px rounded data-end, square at the baseline. */
function barPath(x: number, y: number, w: number, h: number) {
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
}

/** Average overdue balance by prioritisation policy. The model's bar is highlighted. */
export function PolicyChart({ rows }: { rows: PolicyResult[] }) {
  const tip = useTooltip();
  const max = Math.max(...rows.map((r) => r.avg_overdue_balance));
  const base = rows.find((r) => r.policy.startsWith("no proactive"));
  return (
    <figure className="chart policy-chart">
      <ul role="list">
        {rows.map((r) => {
          const isModel = r.policy.startsWith("CashFlow");
          const delta = base ? r.avg_overdue_balance / base.avg_overdue_balance - 1 : 0;
          return (
            <li
              key={r.policy}
              className={isModel ? "policy policy-model" : "policy"}
              onMouseMove={(e) =>
                tip.show(
                  e,
                  <>
                    <strong>{r.policy}</strong>
                    <div>Average overdue balance {money(r.avg_overdue_balance)}</div>
                    <div>DSO {r.dso_days.toFixed(2)} days</div>
                    <div className="tooltip-muted">
                      {r.touches} touches, net savings {money(r.net_savings)}
                    </div>
                  </>,
                )
              }
              onMouseLeave={tip.hide}
            >
              <span className="policy-name">{r.policy}</span>
              <span className="policy-track">
                <span className="policy-fill" style={{ width: `${(r.avg_overdue_balance / max) * 100}%` }} />
              </span>
              <span className="policy-value">
                {moneyCompact(r.avg_overdue_balance)}
                {base && r !== base && <span className="policy-delta"> {(delta * 100).toFixed(0)}%</span>}
              </span>
            </li>
          );
        })}
      </ul>
    </figure>
  );
}

const signed = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(3)}`;

/**
 * SHAP breakdown in log-odds: base value -> invoice score, then each factor's push.
 * The shown factors plus "all other factors" add up exactly to the score.
 */
export function ScoreDrivers({ reasons, base, score }: { reasons: Reason[]; base: number | null; score: number | null }) {
  const tip = useTooltip();
  const overdue = reasons.find((r) => r.feature === "days_overdue");
  const model = reasons.filter((r) => r.feature !== "days_overdue");
  const rest = base !== null && score !== null ? score - base - model.reduce((a, r) => a + r.impact, 0) : null;
  const rows = rest !== null && Math.abs(rest) >= 0.0005 ? [...model, { feature: "_rest", impact: rest, text: "All other factors", direction: rest > 0 ? "raises risk" : "lowers risk" } as Reason] : model;
  const max = Math.max(0.05, ...rows.map((r) => Math.abs(r.impact)));
  return (
    <div className="drivers">
      {base !== null && score !== null && (
        <p className="drivers-line">
          Base <strong>{signed(base)}</strong> <span aria-hidden="true">→</span> invoice score <strong>{signed(score)}</strong>
          <span className="muted"> (log-odds)</span>
        </p>
      )}
      {overdue && (
        <p className="drivers-overdue">
          <span className="status-pill status-critical">Overdue</span> {overdue.text}. The score below is the model's view
          before the due date passed.
        </p>
      )}
      <ul className="driver-list">
        {rows.map((r) => (
          <li
            key={r.feature}
            className={r.feature === "_rest" ? "driver driver-rest" : "driver"}
            onMouseMove={(e) =>
              tip.show(
                e,
                <>
                  <strong>{r.impact > 0 ? "Raises" : "Lowers"} the late-payment score</strong>
                  <div>{signed(r.impact)} log-odds</div>
                </>,
              )
            }
            onMouseLeave={tip.hide}
          >
            <span className="driver-text">{r.text}</span>
            <span className="driver-bar" aria-hidden="true">
              <span className="driver-axis" />
              <span
                className={r.impact > 0 ? "driver-fill up" : "driver-fill down"}
                style={{ width: `${(Math.abs(r.impact) / max) * 50}%` }}
              />
            </span>
            <span className={r.impact > 0 ? "driver-val up" : "driver-val down"}>{signed(r.impact)}</span>
          </li>
        ))}
      </ul>
      <p className="drivers-key muted">
        <span className="key up" /> raises risk <span className="key down" /> lowers risk
      </p>
    </div>
  );
}
