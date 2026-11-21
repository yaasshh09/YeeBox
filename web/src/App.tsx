import { useEffect, useState } from "react";
import { decodeScene, encodeScene, MAX_LINK } from "./share";
import { engineRef, useStore } from "./state/store";
import {
  loadPreset,
  MaterialPicker,
  Oscilloscope,
  RectForm,
  SourceControls,
  StatusStrip,
  ToolRail,
  TopBar,
  Transport,
  useShortcuts,
  ViewControls,
} from "./ui/Panels";
import { Stage } from "./ui/Stage";
import { Verify } from "./ui/Verify";

/** One width threshold for the two shells, as in AtomSIM. */
const NARROW = 900;

function useNarrow() {
  const [narrow, setNarrow] = useState(() => window.innerWidth < NARROW);
  useEffect(() => {
    const on = () => setNarrow(window.innerWidth < NARROW);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return narrow;
}

function Notice() {
  const notice = useStore((s) => s.notice);
  return notice ? (
    <div className="notice" role="status">
      {notice}
    </div>
  ) : null;
}

export default function App() {
  const narrow = useNarrow();
  const [sheet, setSheet] = useState(false);
  const [verify, setVerify] = useState(false);
  useShortcuts();
  useEffect(() => {
    decodeScene(window.location.hash)
      .then((json) => {
        if (json) {
          const s = useStore.getState();
          s.set({ scene: { full: json, compact: null, key: s.scene.key + 1 } });
        } else loadPreset("double_slit");
      })
      .catch((e: Error) => {
        useStore.getState().toast(`That link could not be opened: ${e.message}`);
        loadPreset("double_slit");
      });
  }, []);
  const toast = useStore((s) => s.toast);
  const onShare = async () => {
    const engine = engineRef.current;
    if (!engine) return;
    const json = await engine.call<string>("scene");
    const hash = await encodeScene(json);
    if (hash.length > MAX_LINK) {
      saveFile(json);
      toast("This scene is too detailed for a link, so I saved it as a file. Open it with Open.");
      return;
    }
    history.replaceState(null, "", hash);
    try {
      await navigator.clipboard.writeText(window.location.href);
      toast("Link copied. Anyone who opens it sees this exact scene.");
    } catch {
      toast("The link is in the address bar.");
    }
  };
  const onVerify = () => setVerify(true);

  if (narrow) {
    return (
      <div className="shell mobile">
        <TopBar onShare={onShare} onVerify={onVerify} />
        <main className="center">
          <Stage />
          <StatusStrip />
        </main>
        <div className="mobile-tools">
          <ToolRail />
          <button type="button" className="sheet-toggle" aria-expanded={sheet} onClick={() => setSheet(!sheet)}>
            {sheet ? "Close" : "Controls"}
          </button>
        </div>
        {sheet && (
          <div className="sheet">
            <MaterialPicker />
            <Transport />
            <ViewControls />
            <SourceControls />
            <Oscilloscope />
          </div>
        )}
        <Notice />
        {verify && <Verify onClose={() => setVerify(false)} />}
      </div>
    );
  }
  return (
    <div className="shell">
      <TopBar onShare={onShare} onVerify={onVerify} />
      <div className="grid">
        <aside className="rail">
          <h2 className="section-title">Tools</h2>
          <ToolRail />
          <MaterialPicker />
          <RectForm />
        </aside>
        <main className="center">
          <Stage />
          <StatusStrip />
        </main>
        <aside className="panel">
          <Transport />
          <ViewControls />
          <SourceControls />
          <Oscilloscope />
        </aside>
      </div>
      <Notice />
      {verify && <Verify onClose={() => setVerify(false)} />}
    </div>
  );
}

/** Downloads a scene as a JSON file. */
export function saveFile(json: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  a.download = "yeebox-scene.json";
  a.click();
  URL.revokeObjectURL(a.href);
}
