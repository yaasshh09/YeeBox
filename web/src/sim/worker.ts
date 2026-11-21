// The CPU fallback: a wasm `Sim` built by createCpuSim, stepped and drawn
// here so the page stays responsive. It draws through `pixels()` into the
// OffscreenCanvas the page transferred.
import init, { createCpuSim, type Sim } from "../wasm/fdtd_wasm.js";
import { nextStepsPerFrame, FRAME_BUDGET_MS } from "./pacing";
import type { FromWorker, LoopSettings, ToWorker } from "./protocol";

let sim: Sim | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let settings: LoopSettings = { running: true, maxStepsPerFrame: 8 };
let stepsPerFrame = 1;
let cells = 0;
let window0 = performance.now();
let stepsInWindow = 0;

const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);

function draw() {
  if (!sim || !ctx) return;
  const px = sim.pixels();
  const { width, height } = ctx.canvas;
  if (px.length === width * height * 4) {
    ctx.putImageData(new ImageData(new Uint8ClampedArray(px.buffer as ArrayBuffer, px.byteOffset, px.length), width, height), 0, 0);
  }
}

function tick() {
  if (sim && settings.running) {
    const t0 = performance.now();
    sim.step(stepsPerFrame);
    stepsInWindow += stepsPerFrame;
    draw();
    stepsPerFrame = nextStepsPerFrame(stepsPerFrame, performance.now() - t0, settings.maxStepsPerFrame);
  }
  const now = performance.now();
  if (now - window0 >= 1000) {
    post({ kind: "stats", stats: { stepsPerFrame, mcells: (stepsInWindow * cells) / ((now - window0) * 1e3) } });
    window0 = now;
    stepsInWindow = 0;
  }
  setTimeout(tick, Math.max(0, FRAME_BUDGET_MS - 2));
}

self.onmessage = async (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  try {
    if (m.kind === "init") {
      await init();
      const st = JSON.parse(createCpuSim(m.scene, 1, 1).status()) as { nx: number; ny: number };
      cells = st.nx * st.ny;
      // Painting costs per canvas pixel, and a small grid cannot show more
      // than a few pixels per cell, so the backing store is capped near 2 px
      // per cell and CSS stretches it.
      const scale = Math.max(0.25, Math.min(m.dpr, 2, Math.sqrt((4 * cells) / (m.cssWidth * m.cssHeight))));
      const w = Math.max(1, Math.round(m.cssWidth * scale));
      const h = Math.max(1, Math.round(m.cssHeight * scale));
      sim = createCpuSim(m.scene, w, h);
      m.canvas.width = w;
      m.canvas.height = h;
      ctx = m.canvas.getContext("2d");
      post({ kind: "ready", adapter: sim.adapter(), scale });
      draw();
      tick();
    } else if (m.kind === "loop") {
      settings = m.settings;
    } else if (m.kind === "step") {
      sim?.step(1);
      draw();
    } else if (m.kind === "call") {
      if (!sim) throw new Error("not ready");
      const fn = (sim as unknown as Record<string, (...a: unknown[]) => unknown>)[m.method];
      if (typeof fn !== "function") throw new Error(`no method ${m.method}`);
      const value = fn.apply(sim, m.args);
      if (m.method === "resize" && ctx) {
        ctx.canvas.width = m.args[0] as number;
        ctx.canvas.height = m.args[1] as number;
      }
      if (!settings.running) draw();
      post({ kind: "result", id: m.id, value });
    }
  } catch (e) {
    const err = e as Error & { code?: string };
    if (m.kind === "call") post({ kind: "error", id: m.id, code: err.code ?? "internal", message: err.message });
    else post({ kind: "fatal", code: err.code ?? "internal", message: err.message });
  }
};
