// The controls around the canvas: tools, materials, transport, view, the
// source wavelength, the oscilloscope, a numeric rectangle form, the top bar
// and the status strip. Numbers shown here come from the Rust status; the
// only arithmetic is display formatting.
import { useEffect, useRef, useState } from "react";
import { PRESETS, presetById } from "../presets";
import { engineRef, PALETTE, useStore, type Drive, type Tool } from "../state/store";
import { count, si } from "./format";
import { hintDone } from "./Hints";
import { run } from "./Stage";

const TOOLS: { id: Tool; label: string; key: string }[] = [
  { id: "paint", label: "Paint", key: "p" },
  { id: "erase", label: "Erase", key: "e" },
  { id: "rect", label: "Rectangle", key: "r" },
  { id: "circle", label: "Circle", key: "c" },
  { id: "slits", label: "Slit pair", key: "l" },
  { id: "source", label: "Source", key: "s" },
  { id: "probe", label: "Probe", key: "o" },
  { id: "pan", label: "Pan", key: "h" },
];

export function loadPreset(id: string) {
  const p = presetById(id);
  const s = useStore.getState();
  s.set({ presetId: p.id, scene: { full: p.full, compact: p.compact, key: s.scene.key + 1 } });
}

export function setRunning(running: boolean) {
  useStore.getState().set({ running });
  engineRef.current?.setLoop({ running, maxStepsPerFrame: useStore.getState().speed });
}

/** Keyboard shortcuts for tools and transport. */
export function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === " ") {
        e.preventDefault();
        setRunning(!useStore.getState().running);
      } else if (e.key === ".") engineRef.current?.stepOnce();
      else {
        const tool = TOOLS.find((x) => x.key === e.key.toLowerCase());
        if (tool) useStore.getState().set({ tool: tool.id });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

export function ToolRail() {
  const tool = useStore((s) => s.tool);
  const set = useStore((s) => s.set);
  return (
    <div className="tools" role="toolbar" aria-label="Tools">
      {TOOLS.map((t) => (
        <button
          key={t.id}
          type="button"
          className={tool === t.id ? "tool active" : "tool"}
          aria-pressed={tool === t.id}
          title={`${t.label} (${t.key})`}
          onClick={() => set({ tool: t.id })}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function MaterialPicker() {
  const material = useStore((s) => s.material);
  const brush = useStore((s) => s.brush);
  const set = useStore((s) => s.set);
  return (
    <section>
      <h2 className="section-title">Material</h2>
      <div className="swatches">
        {PALETTE.map((name, k) =>
          k === 0 ? null : (
            <button
              key={name}
              type="button"
              className={material === k ? `swatch m${k} active` : `swatch m${k}`}
              aria-pressed={material === k}
              onClick={() => set({ material: k })}
            >
              {name}
            </button>
          ),
        )}
      </div>
      <label className="row">
        <span>Brush</span>
        <input
          type="range"
          min={0}
          max={12}
          step={0.5}
          value={brush}
          onChange={(e) => set({ brush: Number(e.target.value) })}
        />
        <output>{brush} cells</output>
      </label>
    </section>
  );
}

export function Transport() {
  const running = useStore((s) => s.running);
  const speed = useStore((s) => s.speed);
  const set = useStore((s) => s.set);
  return (
    <section>
      <h2 className="section-title">Transport</h2>
      <div className="buttons">
        <button type="button" onClick={() => setRunning(!running)}>
          {running ? "Pause" : "Play"}
        </button>
        <button type="button" disabled={running} onClick={() => engineRef.current?.stepOnce()}>
          Step
        </button>
        <button type="button" onClick={() => void run((e) => e.call("resetFields"))}>
          Reset fields
        </button>
        <button type="button" onClick={() => void run((e) => e.call("clearScene"))}>
          Clear
        </button>
      </div>
      <label className="row">
        <span>Speed</span>
        <input
          type="range"
          min={1}
          max={64}
          value={speed}
          onChange={(e) => {
            const v = Number(e.target.value);
            set({ speed: v });
            engineRef.current?.setLoop({ running: useStore.getState().running, maxStepsPerFrame: v });
          }}
        />
        <output>{speed} steps per frame</output>
      </label>
    </section>
  );
}

export function ViewControls() {
  const field = useStore((s) => s.field);
  const gain = useStore((s) => s.gain);
  const set = useStore((s) => s.set);
  const apply = (f: "ez" | "intensity", g: number) => {
    set({ field: f, gain: g });
    void run((e) => e.call("setView", JSON.stringify({ field: f, gain: g, auto_gain: false })));
  };
  return (
    <section>
      <h2 className="section-title">View</h2>
      <div className="buttons">
        <button type="button" aria-pressed={field === "ez"} className={field === "ez" ? "active" : ""} onClick={() => apply("ez", gain)}>
          Field
        </button>
        <button
          type="button"
          aria-pressed={field === "intensity"}
          className={field === "intensity" ? "active" : ""}
          onClick={() => apply("intensity", gain)}
        >
          Time-averaged intensity
        </button>
      </div>
      <label className="row">
        <span>Gain</span>
        <input
          type="range"
          min={-3}
          max={3}
          step={0.05}
          value={Math.log2(gain)}
          onChange={(e) => apply(field, 2 ** Number(e.target.value))}
        />
        <output>x{Number(gain.toPrecision(2))}</output>
      </label>
    </section>
  );
}

export function SourceControls() {
  const wavelength = useStore((s) => s.wavelength);
  const status = useStore((s) => s.status);
  const set = useStore((s) => s.set);
  if (wavelength == null) {
    return (
      <section>
        <h2 className="section-title">Source</h2>
        <p className="hint">This scene is lit by a plane wave. Pick the Source tool and tap to add a point source.</p>
      </section>
    );
  }
  const drive = status?.sources.find((d): d is Drive => d != null);
  return (
    <section>
      <h2 className="section-title">Source</h2>
      <label className="row">
        <span>Wavelength</span>
        <input
          type="range"
          min={4}
          max={200}
          value={wavelength}
          onChange={(e) => {
            const v = Number(e.target.value);
            set({ wavelength: v });
            void run((eng) => eng.call("retuneSources", v));
          }}
        />
        <output>{wavelength} cells</output>
      </label>
      {drive && (
        <p className="hint">
          {si(drive.vacuum_wavelength_m, "m")} in vacuum, {si(drive.f_hz, "Hz")}
        </p>
      )}
    </section>
  );
}

export function Oscilloscope() {
  const scope = useStore((s) => s.scope);
  const probeSet = useStore((s) => s.probeSet);
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const { width: w, height: h } = c;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = getComputedStyle(c).getPropertyValue("--edge");
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    if (scope.length < 2) return;
    const peak = scope.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-30);
    ctx.strokeStyle = getComputedStyle(c).getPropertyValue("--text");
    ctx.beginPath();
    scope.forEach((v, k) => {
      const x = (k / (scope.length - 1)) * w;
      const y = h / 2 - (v / peak) * (h / 2 - 2);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [scope]);
  return (
    <section>
      <h2 className="section-title">Oscilloscope</h2>
      <canvas ref={ref} className="scope" width={560} height={160} aria-label="Probe trace" />
      <p className="hint">
        {probeSet ? `Ez at the probe, last ${scope.length} steps, scaled to its peak.` : "Pick the Probe tool and tap the field."}
      </p>
    </section>
  );
}

/** A keyboard way to add a rectangle, for people who cannot drag. */
export function RectForm() {
  const material = useStore((s) => s.material);
  const [v, setV] = useState({ x0: 100, y0: 100, x1: 140, y1: 140 });
  return (
    <section>
      <h2 className="section-title">Add a rectangle</h2>
      <form
        className="rect-form"
        onSubmit={(e) => {
          e.preventDefault();
          void run((eng) => eng.call("addShape", JSON.stringify({ shape: "rect", ...v, material })));
        }}
      >
        {(["x0", "y0", "x1", "y1"] as const).map((k) => (
          <label key={k}>
            <span>{k} (cells)</span>
            <input type="number" value={v[k]} onChange={(e) => setV({ ...v, [k]: Number(e.target.value) })} />
          </label>
        ))}
        <button type="submit">Add</button>
      </form>
    </section>
  );
}

export function TopBar({ onVerify, onShare }: { onVerify: () => void; onShare: () => void }) {
  const presetId = useStore((s) => s.presetId);
  const p = presetById(presetId);
  return (
    <header className="topbar">
      <div className="brand">YeeBox</div>
      <select
        aria-label="Experiment"
        value={presetId}
        onChange={(e) => {
          hintDone("presets");
          loadPreset(e.target.value);
        }}
      >
        {PRESETS.map((x) => (
          <option key={x.id} value={x.id}>
            {x.title}
          </option>
        ))}
      </select>
      <p className="watch">
        {p.watch} <span className="try">{p.tryIt}</span>
        {p.cases.length > 0 && (
          <button type="button" className="badge" onClick={onVerify}>
            matches theory ({p.cases.join(", ")})
          </button>
        )}
      </p>
      <div className="topbar-right">
        <label className="file-button">
          Open
          <input
            type="file"
            accept="application/json,.json"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              const s = useStore.getState();
              s.set({ scene: { full: await file.text(), compact: null, key: s.scene.key + 1 } });
              e.target.value = "";
            }}
          />
        </label>
        <button type="button" onClick={onShare}>
          Share
        </button>
        <button type="button" onClick={onVerify}>
          Verify
        </button>
      </div>
    </header>
  );
}

export function StatusStrip() {
  const status = useStore((s) => s.status);
  const kernel = useStore((s) => s.kernel);
  const adapter = useStore((s) => s.adapter);
  const mcells = useStore((s) => s.mcells);
  const spf = useStore((s) => s.stepsPerFrame);
  if (!status) return <footer className="status">Starting...</footer>;
  const drive = status.plane_wave ?? status.sources.find((d): d is Drive => d != null) ?? null;
  return (
    <footer className="status" aria-live="polite">
      <span>
        {status.nx} x {status.ny} cells
      </span>
      <span>cell {si(status.dx_m, "m")}</span>
      {drive && (
        <>
          <span>
            {si(drive.vacuum_wavelength_m, "m")}, {si(drive.f_hz, "Hz")}
          </span>
          <span className={drive.under_resolved ? "warn" : ""}>
            {drive.cells_per_wavelength.toFixed(1)} cells per wavelength
            {drive.under_resolved ? " (below 10, expect dispersion)" : ""}
          </span>
        </>
      )}
      <span>t {si(status.time_s, "s")}</span>
      <span>
        {count(mcells)} Mcells/s, {spf} steps per frame
      </span>
      <span title={adapter}>{kernel === "gpu" ? "GPU" : "CPU"} kernel</span>
    </footer>
  );
}
