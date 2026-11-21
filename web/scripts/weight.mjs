// First-load page weight: every file the app page can fetch, gzipped,
// against the 2 MB budget in docs/06-experience.md. The self-test page and
// its golden fields are left out; the app never loads them.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const BUDGET = 2 * 1024 * 1024;
const dist = new URL("../dist/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

function walk(dir) {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

let total = 0;
for (const file of walk(dist)) {
  if (/selftest|\.bin$/.test(file)) continue;
  const size = gzipSync(readFileSync(file), { level: 9 }).length;
  total += size;
  console.log(`${(size / 1024).toFixed(1).padStart(8)} kB  ${file.slice(dist.length)}`);
}
console.log(`${(total / 1024).toFixed(1).padStart(8)} kB  total (budget ${BUDGET / 1024} kB)`);
if (total > BUDGET) {
  console.error("over the first-load budget");
  process.exit(1);
}
