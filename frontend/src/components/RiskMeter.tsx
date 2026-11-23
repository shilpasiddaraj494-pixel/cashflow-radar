import { pct } from "../format";
import { useTooltip } from "./Tooltip";

type Props = { p: number; breakeven: number; overdue: boolean; width?: number };

/**
 * The decision rule, drawn: the bar is P(late), the tick is this invoice's break-even
 * probability (touch cost / benefit if late). Bar past the tick = worth a touch.
 */
export function RiskMeter({ p, breakeven, overdue, width = 132 }: Props) {
  const tip = useTooltip();
  const h = 20;
  const barH = 6;
  const y = (h - barH) / 2;
  const fillW = Math.max(2, p * width);
  const tickX = Math.min(width - 1, Math.max(1, breakeven * width));
  const worth = p >= breakeven;
  return (
    <svg
      className="risk-meter"
      width={width}
      height={h}
      role="img"
      aria-label={`${pct(p)} chance of late payment; worth a touch above ${pct(breakeven)}`}
      onMouseMove={(e) =>
        tip.show(
          e,
          <>
            <strong>{overdue ? "Already late" : `${pct(p)} chance of paying late`}</strong>
            <div>Worth a touch above {pct(breakeven)}</div>
            <div className="tooltip-muted">{worth ? "Above break-even: touch" : "Below break-even: monitor"}</div>
          </>,
        )
      }
      onMouseLeave={tip.hide}
    >
      <rect x={0} y={y} width={width} height={barH} rx={3} className="risk-track" />
      <rect x={0} y={y} width={fillW} height={barH} rx={3} className={worth ? "risk-fill risk-fill-worth" : "risk-fill"} />
      <line x1={tickX} x2={tickX} y1={2} y2={h - 2} className="risk-tick" />
    </svg>
  );
}
