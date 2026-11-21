// Steps per frame from frame pacing: back off quickly when frames run long,
// creep up when there is room. Scheduling only; no physics here.

export const FRAME_BUDGET_MS = 1000 / 60;

export function nextStepsPerFrame(current: number, frameMs: number, max: number): number {
  let next = current;
  if (frameMs > FRAME_BUDGET_MS * 1.25) next = Math.floor(current * 0.8);
  else if (frameMs < FRAME_BUDGET_MS * 1.05) next = current + 1;
  return Math.max(1, Math.min(max, next));
}
