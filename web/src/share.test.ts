import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";
import { decodeScene, encodeScene, LINK_PREFIX, MAX_LINK } from "./share";

test("scenes round trip byte for byte through a link", async () => {
  for (const name of ["double_slit", "waveguide", "lens"]) {
    const json = readFileSync(join(__dirname, "presets", `${name}.json`), "utf8");
    const link = await encodeScene(json);
    expect(link.startsWith(LINK_PREFIX)).toBe(true);
    expect(link).toMatch(/^#s=1\.[A-Za-z0-9_-]+$/);
    expect(await decodeScene(link)).toBe(json);
    expect(link.length).toBeLessThan(MAX_LINK);
  }
});

test("no link, newer links and damaged links", async () => {
  expect(await decodeScene("")).toBeNull();
  expect(await decodeScene("#about")).toBeNull();
  await expect(decodeScene("#s=2.abc")).rejects.toThrow(/newer version/);
  await expect(decodeScene("#s=1.@@@")).rejects.toThrow();
});
