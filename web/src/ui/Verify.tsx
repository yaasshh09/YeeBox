// The Verify panel: every acceptance case from the last `fdtd report`, with
// the measured value and gate as Rust formatted them, and links to the full
// report, the in-browser self-test and the source.
import { useEffect } from "react";
import summary from "../../../validation/report/summary.json";

const REPO = "https://github.com/yaasshh09/YeeBox";

export function Verify({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const passed = summary.cases.filter((c) => c.pass).length;
  return (
    <div className="overlay" onClick={onClose}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label="Validation" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-head">
          <h2 className="section-title">Validation</h2>
          <button type="button" onClick={onClose} aria-label="Close">
            Close
          </button>
        </div>
        <p>
          Every frame is a live finite-difference time-domain solve. The GPU kernel is checked against an f64 CPU
          reference, and the reference against analytic physics. {passed} of {summary.cases.length} checks pass.
          CPML reflection: {summary.cpml.map((c) => `${c.worst_db} at ${c.cells} cells`).join(", ")}, worst over 0 to
          70 degrees.
        </p>
        <table className="cases">
          <thead>
            <tr>
              <th>Case</th>
              <th>Check</th>
              <th>Measured</th>
              <th>Gate</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {summary.cases.map((c) => (
              <tr key={c.id}>
                <td>{c.id}</td>
                <td>{c.name}</td>
                <td className="num">{c.measured}</td>
                <td className="num">{c.gate}</td>
                <td className={c.pass ? "pass" : "fail"}>{c.pass ? "pass" : "fail"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="links">
          <a href={`${REPO}/blob/main/validation/report/report.md`} target="_blank" rel="noopener">
            Full report
          </a>
          <a href="selftest.html" target="_blank" rel="noopener">
            Run the GPU self-test in this browser
          </a>
          <a href={REPO} target="_blank" rel="noopener">
            Source
          </a>
        </p>
      </div>
    </div>
  );
}
