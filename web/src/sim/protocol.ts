// Messages between the page and the CPU fallback worker. The worker owns a
// wasm `Sim` (createCpuSim) and its frame loop; the page sends calls by the
// Rust method name and loop settings, and gets results and loop stats back.

export type LoopSettings = {
  running: boolean;
  /** The most steps one frame may take; the loop adapts below it. */
  maxStepsPerFrame: number;
};

export type LoopStats = {
  stepsPerFrame: number;
  /** Million cell updates per second over the last second. */
  mcells: number;
};

export type ToWorker =
  | { kind: "init"; canvas: OffscreenCanvas; scene: string; cssWidth: number; cssHeight: number; dpr: number }
  | { kind: "call"; id: number; method: string; args: unknown[] }
  | { kind: "loop"; settings: LoopSettings }
  | { kind: "step" };

export type FromWorker =
  | { kind: "ready"; adapter: string; scale: number }
  | { kind: "result"; id: number; value: unknown }
  | { kind: "error"; id: number; code: string; message: string }
  | { kind: "stats"; stats: LoopStats }
  | { kind: "fatal"; code: string; message: string };

/** A coded error as the Rust side raises it (`fdtd_wasm::ErrorInfo`). */
export class SimError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

export function toSimError(e: unknown): SimError {
  if (e instanceof SimError) return e;
  if (e instanceof Error) return new SimError((e as Error & { code?: string }).code ?? "internal", e.message);
  return new SimError("internal", String(e));
}
