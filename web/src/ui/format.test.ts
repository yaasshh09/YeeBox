import { expect, test } from "vitest";
import { count, si } from "./format";

test("SI prefixes keep units intact", () => {
  expect(si(2.5e-8, "m")).toBe("25 nm");
  expect(si(5e-7, "m")).toBe("500 nm");
  expect(si(5.99584916e14, "Hz")).toBe("600 THz");
  expect(si(4.99654097e9, "Hz")).toBe("5 GHz");
  expect(si(0.06, "m")).toBe("60 mm");
  expect(si(1.6678e-12, "s")).toBe("1.67 ps");
  expect(si(0, "s")).toBe("0 s");
  expect(si(NaN, "s")).toBe("n/a");
  expect(count(3041.6)).toBe("3,042");
});
