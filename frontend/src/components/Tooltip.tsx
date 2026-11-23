import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

type Tip = { x: number; y: number; content: ReactNode } | null;
type Ctx = { show: (e: { clientX: number; clientY: number }, content: ReactNode) => void; hide: () => void };

const TooltipContext = createContext<Ctx>({ show: () => {}, hide: () => {} });

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<Tip>(null);
  const show = useCallback<Ctx["show"]>((e, content) => setTip({ x: e.clientX, y: e.clientY, content }), []);
  const hide = useCallback(() => setTip(null), []);
  // Flip to the left near the right edge so the tooltip never leaves the viewport.
  const flip = tip && tip.x > window.innerWidth - 260;
  return (
    <TooltipContext.Provider value={{ show, hide }}>
      {children}
      {tip && (
        <div
          className="tooltip"
          role="tooltip"
          style={{ left: flip ? tip.x - 14 : tip.x + 14, top: tip.y + 14, transform: flip ? "translateX(-100%)" : undefined }}
        >
          {tip.content}
        </div>
      )}
    </TooltipContext.Provider>
  );
}

export const useTooltip = () => useContext(TooltipContext);
