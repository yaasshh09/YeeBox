// The in-browser self-test page: the browser's GPU kernel against the native
// CPU reference fields (validation/golden/selftest), through runSelfTest.
import "./index.css";
import init, { runSelfTest } from "./wasm/fdtd_wasm.js";
import cavity from "../../validation/scenes/cavity_tm.json?raw";
import painted from "../../validation/scenes/painted_walls.json?raw";
import rod from "../../validation/scenes/tfsf_rod_minus_x.json?raw";
import cavityGold from "../../validation/golden/selftest/cavity_tm.bin?url";
import paintedGold from "../../validation/golden/selftest/painted_walls.bin?url";
import rodGold from "../../validation/golden/selftest/tfsf_rod_minus_x.bin?url";

type Report = {
  scene: string;
  adapter: string;
  gate: number;
  diffs: { step: number; fields: number[]; worst: number }[];
  worst: number;
  pass: boolean;
};

const SCENES: [string, string, string][] = [
  ["cavity_tm", cavity, cavityGold],
  ["painted_walls", painted, paintedGold],
  ["tfsf_rod_minus_x", rod, rodGold],
];

const sci = (v: number) => v.toExponential(2);
const body = document.querySelector("#results tbody")!;
const verdict = document.getElementById("verdict")!;
const adapter = document.getElementById("adapter")!;

async function main() {
  await init();
  let all = true;
  for (const [name, json, gold] of SCENES) {
    const bytes = new Uint8Array(await (await fetch(gold)).arrayBuffer());
    const r = JSON.parse(await runSelfTest(name, json, bytes)) as Report;
    adapter.textContent = `GPU: ${r.adapter}`;
    for (const d of r.diffs) {
      const tr = document.createElement("tr");
      const ok = d.worst < r.gate;
      tr.innerHTML = `<td>${name}</td><td>${d.step}</td>${d.fields
        .map((f) => `<td class="num">${sci(f)}</td>`)
        .join("")}<td class="${ok ? "pass" : "fail"}">${ok ? "pass" : "fail"}</td>`;
      body.appendChild(tr);
    }
    all &&= r.pass;
  }
  verdict.textContent = all
    ? "Every field of every scene is within 1e-4 of the reference peak. This browser's GPU kernel matches the native CPU kernel."
    : "At least one field is outside the gate. Please report your browser and GPU.";
  verdict.className = all ? "pass" : "fail";
  document.body.dataset.result = all ? "pass" : "fail";
}

main().catch((e: Error & { code?: string }) => {
  adapter.textContent =
    e.code === "no_webgpu" ? "This browser has no WebGPU, so there is no GPU kernel to test." : `Error: ${e.message}`;
  document.body.dataset.result = e.code === "no_webgpu" ? "skipped" : "error";
});
