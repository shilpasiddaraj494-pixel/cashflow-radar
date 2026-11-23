import { useCallback, useState } from "react";

const KEY = "cfr.contacted";

function read(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(KEY) ?? "[]"));
  } catch {
    return new Set();
  }
}

/** Invoices the collector has marked as contacted. Kept in this browser only. */
export function useContacted() {
  const [ids, setIds] = useState<Set<string>>(read);
  const toggle = useCallback((id: string) => {
    setIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      try {
        localStorage.setItem(KEY, JSON.stringify([...next]));
      } catch {
        /* storage unavailable: keep in memory */
      }
      return next;
    });
  }, []);
  return { has: (id: string) => ids.has(id), toggle };
}
