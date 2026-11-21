// Three one-line hints, each shown once until its action is first done,
// remembered across visits. No tutorial, no modal.
import { useEffect, useState } from "react";
import { useStore } from "../state/store";

export type HintId = "draw" | "source" | "presets";

const TEXT: Record<HintId, string> = {
  draw: "Drag on the field to draw a wall.",
  source: "Pick Source and tap to add another source.",
  presets: "Try another experiment from the menu at the top.",
};

const KEY = "yeebox.hints.done";

function load(): HintId[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "[]") as HintId[];
  } catch {
    return [];
  }
}

/** Marks a hint's action as done, for good. */
export function hintDone(id: HintId) {
  const done = load();
  if (!done.includes(id)) {
    localStorage.setItem(KEY, JSON.stringify([...done, id]));
    window.dispatchEvent(new Event("yeebox-hints"));
  }
}

export function Hints() {
  const [done, setDone] = useState(load);
  const kernel = useStore((s) => s.kernel);
  useEffect(() => {
    const on = () => setDone(load());
    window.addEventListener("yeebox-hints", on);
    return () => window.removeEventListener("yeebox-hints", on);
  }, []);
  const next = (["draw", "source", "presets"] as HintId[]).find((h) => !done.includes(h));
  return (
    <>
      {kernel === "cpu" && (
        <p className="cpu-notice" role="note">
          This browser has no WebGPU, so a smaller version runs on the CPU.{" "}
          <a href="https://caniuse.com/webgpu" target="_blank" rel="noopener">
            Which browsers run the full version
          </a>
        </p>
      )}
      {next && kernel && <p className="hint-bubble">{TEXT[next]}</p>}
    </>
  );
}
