import { expect, test } from "vitest";
import { nextStepsPerFrame } from "./pacing";

test("backs off on long frames and creeps up on short ones", () => {
  expect(nextStepsPerFrame(10, 30, 48)).toBe(8);
  expect(nextStepsPerFrame(10, 10, 48)).toBe(11);
  expect(nextStepsPerFrame(48, 10, 48)).toBe(48);
  expect(nextStepsPerFrame(1, 100, 48)).toBe(1);
  expect(nextStepsPerFrame(20, 19, 48)).toBe(20);
});
