// `text-transform: uppercase` turns "mm" into "MM" and "GHz" into "GHZ",
// which corrupts units and scientific notation. CLAUDE.md allows it on
// section headings only; this keeps every such rule on `.section-title`.
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

function cssFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? cssFiles(join(dir, e.name)) : e.name.endsWith(".css") ? [join(dir, e.name)] : [],
  );
}

test("uppercase only on section headings", () => {
  const files = cssFiles(join(__dirname));
  expect(files.length).toBeGreaterThan(0);
  let rules = 0;
  for (const file of files) {
    const css = readFileSync(file, "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
    for (const m of css.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
      if (/text-transform\s*:\s*uppercase/.test(m[2])) {
        rules++;
        const selectors = m[1].split(",").map((s) => s.trim());
        for (const s of selectors) expect(s, `${file}: ${s}`).toBe(".section-title");
      }
    }
  }
  expect(rules).toBeGreaterThan(0);
});
