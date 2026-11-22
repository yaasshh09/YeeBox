// The polar plot: the far-field pattern Rust computes from the scene's
// contour (Sim.farField), in dB below its peak, with a dashed theory curve
// when there is one: the live array factor from Rust, or the preset's stored
// curve while the scene is still exactly as shipped. Only drawing here.
import { presetById } from "../presets";
import { REFERENCES } from "../presets/references";
import { useStore } from "../state/store";
import { run } from "./Stage";

const SIZE = 240;
const R = 104;
/** The plot spans 0 to -30 dB from the rim to the center. */
const FLOOR = -30;

/** An SVG path through one value per direction, angles from +x, counterclockwise. */
function path(db: number[]): string {
  const c = SIZE / 2;
  return (
    db
      .map((v, m) => {
        const r = (R * (Math.max(v, FLOOR) - FLOOR)) / -FLOOR;
        const a = (2 * Math.PI * m) / db.length;
        return `${m === 0 ? "M" : "L"}${(c + r * Math.cos(a)).toFixed(1)},${(c - r * Math.sin(a)).toFixed(1)}`;
      })
      .join(" ") + " Z"
  );
}

export function PolarPlot() {
  const far = useStore((s) => s.far);
  const presetId = useStore((s) => s.presetId);
  const kernel = useStore((s) => s.kernel);
  const steerAngle = useStore((s) => s.steer);
  const set = useStore((s) => s.set);
  if (!far) return null;
  const preset = presetById(presetId);
  const stored = preset.id === presetId && far.edits === 0 ? REFERENCES[presetId] : undefined;
  const theory = far.theory ?? (stored ? stored[kernel === "gpu" ? "full" : "compact"] : null);
  const label = far.theory ? "Array factor" : stored?.label;
  const c = SIZE / 2;
  return (
    <section>
      <h2 className="section-title">Far field</h2>
      <svg className="polar" viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Far-field pattern">
        {[0, -10, -20].map((db) => (
          <circle key={db} cx={c} cy={c} r={(R * (db - FLOOR)) / -FLOOR} className="polar-ring" />
        ))}
        <line x1={c - R} y1={c} x2={c + R} y2={c} className="polar-ring" />
        <line x1={c} y1={c - R} x2={c} y2={c + R} className="polar-ring" />
        {theory && <path d={path(theory)} className="polar-theory" />}
        <path d={path(far.db)} className="polar-sim" />
      </svg>
      {preset.steer && preset.id === presetId && (
        <label className="row">
          <span>Steer</span>
          <input
            type="range"
            min={-60}
            max={60}
            value={steerAngle}
            onChange={(e) => {
              const v = Number(e.target.value);
              set({ steer: v });
              void run((eng) => eng.call("steer", v));
            }}
          />
          <output>{steerAngle} deg</output>
        </label>
      )}
      <p className="hint">
        Pattern in dB below its peak, rings every 10 dB, +x to the right. It starts over after every edit and
        settles in a few periods.
        {label ? ` Dashed: ${label.toLowerCase()}.` : ""}
      </p>
    </section>
  );
}
