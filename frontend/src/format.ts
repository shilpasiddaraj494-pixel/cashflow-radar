const eur0 = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });

export function money(x: number): string {
  return eur0.format(x);
}

/** Compact money for headline numbers: €8.7M, €412k, €950. */
export function moneyCompact(x: number): string {
  const a = Math.abs(x);
  const sign = x < 0 ? "−" : "";
  if (a >= 1e6) return `${sign}€${(a / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e4) return `${sign}€${Math.round(a / 1e3)}k`;
  if (a >= 1e3) return `${sign}€${(a / 1e3).toFixed(1)}k`;
  return `${sign}€${Math.round(a)}`;
}

export function pct(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`;
}

export function days(d: number): string {
  const r = Math.round(d);
  return `${r} day${Math.abs(r) === 1 ? "" : "s"}`;
}

const dateFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" });
const shortDate = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short" });

export function date(iso: string): string {
  return dateFmt.format(new Date(iso));
}

export function dateShort(iso: string): string {
  return shortDate.format(new Date(iso));
}

/** "due in 5 days", "due today", "12 days overdue" */
export function dueLabel(daysToDue: number): string {
  if (daysToDue > 0) return `due in ${days(daysToDue)}`;
  if (daysToDue === 0) return "due today";
  return `${days(-daysToDue)} overdue`;
}
