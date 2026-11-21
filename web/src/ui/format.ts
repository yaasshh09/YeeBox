// Display formatting only: SI prefixes for the status strip.

const PREFIXES: [number, string][] = [
  [1e12, "T"],
  [1e9, "G"],
  [1e6, "M"],
  [1e3, "k"],
  [1, ""],
  [1e-3, "m"],
  [1e-6, "µ"],
  [1e-9, "n"],
  [1e-12, "p"],
  [1e-15, "f"],
];

/** `value` in `unit` with an SI prefix and three significant digits. */
export function si(value: number, unit: string): string {
  if (!Number.isFinite(value)) return "n/a";
  if (value === 0) return `0 ${unit}`;
  const a = Math.abs(value);
  const [scale, prefix] = PREFIXES.find(([s]) => a >= s * 0.9995) ?? PREFIXES[PREFIXES.length - 1];
  return `${Number((value / scale).toPrecision(3))} ${prefix}${unit}`;
}

/** A count with thousands separators. */
export function count(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}
