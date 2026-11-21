// The one object the UI talks to. It picks the kernel (WebGPU on the main
// thread, else the f64 CPU fallback in a worker), owns the frame loop, and
// forwards calls to the wasm `Sim` by method name. No numbers are computed
// here except frame timing.
import init, { createSim, type Sim } from "../wasm/fdtd_wasm.js";
import { nextStepsPerFrame } from "./pacing";
import { SimError, toSimError, type FromWorker, type LoopSettings, type LoopStats, type ToWorker } from "./protocol";

export type Kernel = "gpu" | "cpu";

export type EngineEvent =
  | { kind: "stats"; stats: LoopStats }
  | { kind: "lost"; message: string; scene: string | null }
  | { kind: "error"; error: SimError };

export interface Engine {
  readonly kernel: Kernel;
  readonly adapter: string;
  readonly canvas: HTMLCanvasElement;
  /** Calls a `Sim` method by its JS name; resolves with its result. */
  call<T = unknown>(method: string, ...args: unknown[]): Promise<T>;
  setLoop(settings: LoopSettings): void;
  /** One step while paused. */
  stepOnce(): void;
  dispose(): void;
}

/** Device pixels for a CSS size, at most 2x, so phones do not pay 3x. */
export function devicePixels(css: number): number {
  return Math.max(1, Math.round(css * Math.min(window.devicePixelRatio || 1, 2)));
}

function makeCanvas(host: HTMLElement): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.className = "stage-canvas";
  host.replaceChildren(c);
  c.width = devicePixels(host.clientWidth);
  c.height = devicePixels(host.clientHeight);
  return c;
}

async function hasAdapter(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return (await gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/**
 * Builds an engine in `host`. `scene` is used on the GPU, `cpuScene` (a
 * smaller version, if given) on the CPU fallback.
 */
export async function createEngine(
  host: HTMLElement,
  scene: string,
  cpuScene: string | null,
  settings: LoopSettings,
  onEvent: (e: EngineEvent) => void,
  forceCpu = false,
): Promise<Engine> {
  if (!forceCpu && (await hasAdapter())) {
    const canvas = makeCanvas(host);
    try {
      await init();
      const sim = await createSim(canvas, scene);
      return gpuEngine(canvas, sim, settings, onEvent);
    } catch (e) {
      const err = toSimError(e);
      if (err.code !== "no_webgpu") throw err;
    }
  }
  return cpuEngine(makeCanvas(host), cpuScene ?? scene, settings, onEvent);
}

function gpuEngine(
  canvas: HTMLCanvasElement,
  sim: Sim,
  settings: LoopSettings,
  onEvent: (e: EngineEvent) => void,
): Engine {
  let loop = settings;
  let steps = Math.min(4, loop.maxStepsPerFrame);
  let last = performance.now();
  let window0 = last;
  let stepsInWindow = 0;
  const cells = (() => {
    const st = JSON.parse(sim.status()) as { nx: number; ny: number };
    return st.nx * st.ny;
  })();
  let frame = 0;
  let disposed = false;

  const lose = (e: unknown) => {
    if (disposed) return;
    disposed = true;
    cancelAnimationFrame(frame);
    let scene: string | null = null;
    try {
      scene = sim.scene();
    } catch {
      scene = null;
    }
    onEvent({ kind: "lost", message: toSimError(e).message, scene });
  };

  const tick = (now: number) => {
    if (disposed) return;
    try {
      if (loop.running) {
        steps = nextStepsPerFrame(steps, now - last, loop.maxStepsPerFrame);
        sim.step(steps);
        stepsInWindow += steps;
      }
      sim.check();
      sim.render();
    } catch (e) {
      lose(e);
      return;
    }
    last = now;
    if (now - window0 >= 1000) {
      onEvent({ kind: "stats", stats: { stepsPerFrame: steps, mcells: (stepsInWindow * cells) / ((now - window0) * 1e3) } });
      window0 = now;
      stepsInWindow = 0;
    }
    frame = requestAnimationFrame(tick);
  };
  frame = requestAnimationFrame(tick);

  return {
    kernel: "gpu",
    adapter: sim.adapter(),
    canvas,
    call<T>(method: string, ...args: unknown[]): Promise<T> {
      try {
        const fn = (sim as unknown as Record<string, (...a: unknown[]) => unknown>)[method];
        if (typeof fn !== "function") throw new SimError("internal", `no method ${method}`);
        return Promise.resolve(fn.apply(sim, args) as T);
      } catch (e) {
        return Promise.reject(toSimError(e));
      }
    },
    setLoop(s) {
      loop = s;
      steps = Math.min(steps, s.maxStepsPerFrame);
    },
    stepOnce() {
      sim.step(1);
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      sim.free();
    },
  };
}

function cpuEngine(
  canvas: HTMLCanvasElement,
  scene: string,
  settings: LoopSettings,
  onEvent: (e: EngineEvent) => void,
): Promise<Engine> {
  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: SimError) => void }>();
  let nextId = 1;
  const send = (m: ToWorker, transfer: Transferable[] = []) => worker.postMessage(m, transfer);
  return new Promise((resolve, reject) => {
    worker.onmessage = (ev: MessageEvent<FromWorker>) => {
      const m = ev.data;
      if (m.kind === "ready") {
        send({ kind: "loop", settings });
        resolve({
          kernel: "cpu",
          adapter: m.adapter,
          canvas,
          call<T>(method: string, ...args: unknown[]): Promise<T> {
            const id = nextId++;
            send({ kind: "call", id, method, args });
            return new Promise<T>((res, rej) => pending.set(id, { resolve: res as (v: unknown) => void, reject: rej }));
          },
          setLoop(s) {
            send({ kind: "loop", settings: s });
          },
          stepOnce() {
            send({ kind: "step" });
          },
          dispose() {
            worker.terminate();
          },
        });
      } else if (m.kind === "result" || m.kind === "error") {
        const p = pending.get(m.id);
        pending.delete(m.id);
        if (m.kind === "result") p?.resolve(m.value);
        else p?.reject(new SimError(m.code, m.message));
      } else if (m.kind === "stats") {
        onEvent({ kind: "stats", stats: m.stats });
      } else if (m.kind === "fatal") {
        const err = new SimError(m.code, m.message);
        onEvent({ kind: "error", error: err });
        reject(err);
      }
    };
    const offscreen = canvas.transferControlToOffscreen();
    send({ kind: "init", canvas: offscreen, scene, width: canvas.width, height: canvas.height }, [offscreen]);
  });
}
