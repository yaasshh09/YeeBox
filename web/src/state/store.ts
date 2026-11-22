// UI state. The engine itself lives outside the store (engineRef) so frames
// never cause renders; the store holds what the controls show.
import { create } from "zustand";
import type { Engine, Kernel } from "../sim/engine";

export type Tool = "paint" | "erase" | "rect" | "circle" | "slits" | "source" | "probe" | "pan";

export type Drive = {
  f_hz: number;
  vacuum_wavelength_m: number;
  cells_per_wavelength: number;
  under_resolved: boolean;
};

export type Status = {
  nx: number;
  ny: number;
  dx_m: number;
  steps: number;
  time_s: number;
  sources: (Drive | null)[];
  plane_wave: Drive | null;
  /** Resolution warnings for dispersive materials in use (`scene::Warning`). */
  warnings: Warning[];
  mode: Mode;
};

export type Mode = "tm" | "te";

export type Warning =
  | { kind: "skin_depth"; material: number; cells: number }
  | { kind: "material_wavelength"; material: number; cells: number };

/** `Sim.farField`: the pattern of the scene's first contour. */
export type FarField = {
  freq: number;
  angles: number;
  /** dB below the peak at each angle from +x, counterclockwise, floored at -60. */
  db: number[];
  /** The array factor the same way when the scene is a plain CW array. */
  theory: number[] | null;
  /** Scene edits since loading. */
  edits: number;
};

/** The scene to load and the compact twin for the CPU fallback. */
export type SceneSource = { full: string; compact: string | null; key: number };

export const engineRef: { current: Engine | null } = { current: null };

/** Palette order, as `material::palette` in fdtd-core. */
export const PALETTE = ["Vacuum", "Glass", "High index", "Conductor", "Absorber", "Silver"] as const;

/** Palette index of silver, the one dispersive paint. */
export const SILVER = 5;

const reducedMotion =
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

type State = {
  scene: SceneSource;
  presetId: string;
  kernel: Kernel | null;
  adapter: string;
  tool: Tool;
  material: number;
  brush: number;
  running: boolean;
  speed: number;
  field: "ez" | "intensity";
  gain: number;
  wavelength: number | null;
  status: Status | null;
  stepsPerFrame: number;
  mcells: number;
  notice: string | null;
  scope: number[];
  probeSet: boolean;
  far: FarField | null;
  /** The steering slider of an array preset, degrees from +x. */
  steer: number;
  set: (p: Partial<State>) => void;
  toast: (msg: string) => void;
};

export const SCOPE_LEN = 1024;

export const useStore = create<State>((set) => ({
  scene: { full: "", compact: null, key: 0 },
  presetId: "double_slit",
  kernel: null,
  adapter: "",
  tool: "paint",
  material: 3,
  brush: 2,
  running: !reducedMotion,
  speed: 16,
  field: "ez",
  gain: 1,
  wavelength: null,
  status: null,
  stepsPerFrame: 0,
  mcells: 0,
  notice: null,
  scope: [],
  probeSet: false,
  far: null,
  steer: 20,
  set: (p) => set(p),
  toast: (msg) => {
    set({ notice: msg });
    setTimeout(() => set((s) => (s.notice === msg ? { notice: null } : {})), 4000);
  },
}));
