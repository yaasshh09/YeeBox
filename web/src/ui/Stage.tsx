// The canvas: builds the engine for the current scene, polls status and the
// probe, and turns pointer input into tool calls. Positions are converted
// to cells by the Rust viewport (cellAt); TypeScript only snaps them.
import { useEffect, useRef } from "react";
import { createEngine, type Engine } from "../sim/engine";
import { toSimError } from "../sim/protocol";
import { engineRef, SCOPE_LEN, SILVER, useStore } from "../state/store";

// When the last "silver kept clear of the border" note was shown.
let lastSkipNote = -Infinity;
import { hintDone } from "./Hints";

type Pt = { x: number; y: number };

/** Runs engine calls one after another, so strokes land in order. */
let queue: Promise<unknown> = Promise.resolve();
export function run<T>(f: (e: Engine) => Promise<T>): Promise<T | undefined> {
  const next = queue.then(() => (engineRef.current ? f(engineRef.current) : undefined));
  queue = next.catch((e) => useStore.getState().toast(toSimError(e).message));
  return next;
}

function readScene(json: string) {
  const s = JSON.parse(json) as {
    view?: { field: "ez" | "intensity"; gain: number };
    sources?: { waveform: { f0?: number; f_peak?: number } }[];
  };
  const f = s.sources?.map((x) => x.waveform.f0 ?? x.waveform.f_peak).find((v) => v != null);
  return { view: s.view, wavelength: f ? Math.round(1 / f) : null };
}

export function Stage() {
  const host = useRef<HTMLDivElement>(null);
  const scene = useStore((s) => s.scene);

  // Engine lifecycle: one per scene load.
  useEffect(() => {
    const el = host.current;
    if (!el || !scene.full) return;
    let alive = true;
    let engine: Engine | null = null;
    const st = useStore.getState();
    const loop = { running: st.running, maxStepsPerFrame: st.speed };
    const build = async (full: string, compact: string | null) => {
      try {
        engine = await createEngine(el, full, compact, loop, (ev) => {
          if (ev.kind === "stats") useStore.getState().set(ev.stats);
          else if (ev.kind === "lost") {
            useStore.getState().toast("The GPU was reset, so I restored the scene.");
            engine?.dispose();
            if (ev.scene && alive) void build(ev.scene, null);
          } else useStore.getState().toast(ev.error.message);
        });
        if (!alive) {
          engine.dispose();
          return;
        }
        engineRef.current = engine;
        // Cold-load timing: navigation start to a running engine.
        if (!performance.getEntriesByName("yeebox-ready").length) performance.mark("yeebox-ready");
        // A handle for end-to-end tests and the curious: window.yeebox.call("scene").
        (window as unknown as { yeebox: Engine }).yeebox = engine;
        const { view, wavelength } = readScene(full);
        useStore.getState().set({
          kernel: engine.kernel,
          adapter: engine.adapter,
          field: view?.field ?? "ez",
          gain: view?.gain ?? 1,
          wavelength,
          scope: [],
          probeSet: false,
          far: null,
          steer: 20,
        });
      } catch (e) {
        useStore.getState().toast(toSimError(e).message);
      }
    };
    void build(scene.full, scene.compact);
    const poll = setInterval(async () => {
      const e = engineRef.current;
      if (!e) return;
      try {
        const status = JSON.parse(await e.call<string>("status"));
        const samples = await e.call<Float32Array>("takeProbe");
        const far = JSON.parse(await e.call<string>("farField"));
        const s = useStore.getState();
        const scope = samples.length ? [...s.scope, ...samples].slice(-SCOPE_LEN) : s.scope;
        s.set({ status, scope, far });
      } catch {
        // The next poll tries again.
      }
    }, 250);
    const resize = new ResizeObserver(() => {
      const e = engineRef.current;
      if (!e) return;
      const w = Math.max(1, Math.round(el.clientWidth * e.pixelScale));
      const h = Math.max(1, Math.round(el.clientHeight * e.pixelScale));
      if (e.kernel === "gpu") {
        e.canvas.width = w;
        e.canvas.height = h;
      }
      void run((eng) => eng.call("resize", w, h));
    });
    resize.observe(el);
    return () => {
      alive = false;
      clearInterval(poll);
      resize.disconnect();
      engine?.dispose();
      engineRef.current = null;
    };
  }, [scene]);

  // Pointer input.
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const pointers = new Map<number, Pt>();
    let last: Pt | null = null;
    let start: Pt | null = null;
    let dragSource: number | null = null;
    let pinch: { d: number; c: Pt } | null = null;
    const dev = (e: PointerEvent | WheelEvent): Pt => {
      const r = el.getBoundingClientRect();
      const k = engineRef.current?.pixelScale ?? 1;
      return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k };
    };
    const cell = (e: Engine, p: Pt) => e.call<Float64Array>("cellAt", p.x, p.y);
    const spread = () => {
      const [a, b] = [...pointers.values()];
      return { d: Math.hypot(a.x - b.x, a.y - b.y), c: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } };
    };
    const paint = (from: Pt, to: Pt) => {
      hintDone("draw");
      const s = useStore.getState();
      const mat = s.tool === "erase" ? 0 : s.material;
      void run(async (e) => {
        const a = await cell(e, from);
        const b = await cell(e, to);
        const out = JSON.parse(await e.call<string>("paintStroke", a[0], a[1], b[0], b[1], s.brush, mat));
        // Silver may not touch the absorbing border; say so, once in a while.
        if (mat === SILVER && out.skipped > 0 && performance.now() - lastSkipNote > 4000) {
          lastSkipNote = performance.now();
          useStore.getState().toast("Metal stays two cells clear of the absorbing border, so a few cells were left as they were.");
        }
      });
    };

    const down = (ev: PointerEvent) => {
      try {
        el.setPointerCapture(ev.pointerId);
      } catch {
        // Synthetic or already-released pointers cannot be captured.
      }
      const p = dev(ev);
      pointers.set(ev.pointerId, p);
      if (pointers.size === 2) {
        pinch = spread();
        last = null;
        start = null;
        return;
      }
      if (pointers.size > 2) return;
      const s = useStore.getState();
      last = p;
      start = p;
      if (s.tool === "paint" || s.tool === "erase") paint(p, p);
      else if (s.tool === "source") {
        void run(async (e) => {
          const c = await cell(e, p);
          const hit = await e.call<number | undefined>("sourceNear", c[0], c[1], 4);
          if (hit != null) dragSource = hit;
          else {
            await e.call("addSource", c[0], c[1], s.wavelength ?? 20);
            hintDone("source");
          }
        });
      } else if (s.tool === "probe") {
        void run(async (e) => {
          const c = await cell(e, p);
          await e.call("setProbe", c[0], c[1]);
          useStore.getState().set({ probeSet: true, scope: [] });
        });
      } else if (s.tool === "slits") {
        void run(async (e) => {
          const c = await cell(e, p);
          const st = useStore.getState().status;
          if (!st) return;
          const wl = s.wavelength ?? 20;
          const spec = {
            x: Math.round(c[0]),
            thickness: 4,
            y0: 0,
            y1: st.ny - 1,
            center: c[1],
            separation: Math.max(8, wl * 4),
            width: Math.max(3, wl * 0.8),
            material: s.material === 0 ? 3 : s.material,
          };
          await e.call("addSlitPair", JSON.stringify(spec));
        });
      }
    };

    const move = (ev: PointerEvent) => {
      if (!pointers.has(ev.pointerId)) return;
      const p = dev(ev);
      pointers.set(ev.pointerId, p);
      if (pinch && pointers.size === 2) {
        const now = spread();
        const prev = pinch;
        pinch = now;
        void run(async (e) => {
          await e.call("pan", now.c.x - prev.c.x, now.c.y - prev.c.y);
          if (prev.d > 0) await e.call("zoomAt", now.c.x, now.c.y, now.d / prev.d);
        });
        return;
      }
      if (!last) return;
      const s = useStore.getState();
      const from = last;
      last = p;
      if (s.tool === "paint" || s.tool === "erase") paint(from, p);
      else if (s.tool === "pan") void run((e) => e.call("pan", p.x - from.x, p.y - from.y));
      else if (s.tool === "source" && dragSource != null) {
        const k = dragSource;
        void run(async (e) => {
          const c = await cell(e, p);
          return e.call("editSource", k, JSON.stringify({ i: Math.round(c[0]), j: Math.round(c[1]) }));
        });
      }
    };

    const up = (ev: PointerEvent) => {
      pointers.delete(ev.pointerId);
      if (pointers.size < 2) pinch = null;
      const s = useStore.getState();
      const p = dev(ev);
      if (start && (s.tool === "rect" || s.tool === "circle")) {
        const a = start;
        const mat = s.material;
        const kind = s.tool;
        void run(async (e) => {
          const ca = await cell(e, a);
          const cb = await cell(e, p);
          return e.call("addDragShape", kind, ca[0], ca[1], cb[0], cb[1], mat);
        });
      }
      last = null;
      start = null;
      dragSource = null;
    };

    const wheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const p = dev(ev);
      void run((e) => e.call("zoomAt", p.x, p.y, Math.exp(-ev.deltaY * 0.0015)));
    };

    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("wheel", wheel, { passive: false });
    return () => {
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      el.removeEventListener("wheel", wheel);
    };
  }, []);

  return <div ref={host} className="stage" role="application" aria-label="Simulation canvas" />;
}
