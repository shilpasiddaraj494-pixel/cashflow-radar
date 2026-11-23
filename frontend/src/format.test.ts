import { describe, expect, it } from "vitest";
import { dueLabel, moneyCompact, pct } from "./format";

describe("format", () => {
  it("compacts money", () => {
    expect(moneyCompact(8_663_402)).toBe("€8.7M");
    expect(moneyCompact(412_300)).toBe("€412k");
    expect(moneyCompact(1_250)).toBe("€1.3k");
    expect(moneyCompact(-950)).toBe("−€950");
  });
  it("labels due dates", () => {
    expect(dueLabel(5)).toBe("due in 5 days");
    expect(dueLabel(0)).toBe("due today");
    expect(dueLabel(-1)).toBe("1 day overdue");
  });
  it("formats percentages", () => {
    expect(pct(0.573)).toBe("57%");
  });
});
