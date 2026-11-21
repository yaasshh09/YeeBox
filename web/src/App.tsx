import { useEffect, useState } from "react";
import { useStore } from "./state/store";
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
  useShortcuts();
  useEffect(() => {
    if (!useStore.getState().scene.full) loadPreset("double_slit");
  }, []);
  const toast = useStore((s) => s.toast);
  const onShare = () => toast("Sharing arrives in the next build.");
  const onVerify = () => toast("The validation summary arrives in the next build.");

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
    </div>
  );
}
