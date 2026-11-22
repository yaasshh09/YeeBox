import { expect, test, type Page } from "@playwright/test";

type Yee = { call<T>(m: string, ...a: unknown[]): Promise<T> };

async function ready(page: Page) {
  await page.waitForFunction(() => "yeebox" in window, null, { timeout: 30_000 });
  await expect(page.locator(".status")).toContainText("cells", { timeout: 30_000 });
}

const scene = (page: Page) => page.evaluate(() => (window as unknown as { yeebox: Yee }).yeebox.call<string>("scene"));

test("loads, runs, and says it is on the CPU fallback", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await expect(page.locator(".cpu-notice")).toContainText("no WebGPU");
  await expect(page.locator(".status")).toContainText("CPU kernel");
  const time = () => page.locator(".status").innerText();
  const before = await time();
  await page.waitForTimeout(1500);
  expect(await time()).not.toBe(before);
});

test("a drag paints a wall", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  expect(JSON.parse(await scene(page)).paint).toBeUndefined();
  const box = (await page.locator(".stage").boundingBox())!;
  const x = box.x + box.width * 0.5;
  await page.mouse.move(x, box.y + box.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(x, box.y + box.height * 0.7, { steps: 10 });
  await page.mouse.up();
  await expect.poll(async () => JSON.parse(await scene(page)).paint?.runs.length ?? 0).toBeGreaterThan(1);
});

test("a share link reproduces the scene exactly", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");
  await ready(page);
  const box = (await page.locator(".stage").boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.6, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => JSON.parse(await scene(page)).paint?.runs.length ?? 0).toBeGreaterThan(1);
  const shared = await scene(page);
  await page.getByRole("button", { name: "Share" }).click();
  await expect(page).toHaveURL(/#s=1\./);
  const url = page.url();
  const other = await context.newPage();
  await other.goto(url);
  await ready(other);
  expect(await scene(other)).toBe(shared);
});

test("the plasmon preset runs in TE with its badge on the A21 plot", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await page.getByLabel("Experiment").selectOption("plasmon");
  await expect(page.locator(".status")).toContainText("TE");
  await expect.poll(async () => JSON.parse(await scene(page)).mode).toBe("te");
  await expect(page.locator("a.badge")).toHaveAttribute("href", /drude_cylinder/);
});

test("switching the plasmon to TM rebuilds it and drops the preset", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await page.getByLabel("Experiment").selectOption("plasmon");
  await expect(page.locator(".status")).toContainText("TE");
  await page.getByRole("button", { name: "TM (E out of plane)" }).click();
  await expect(page.locator(".status")).toContainText("TM");
  await expect.poll(async () => JSON.parse(await scene(page)).mode).toBeUndefined();
  await expect(page.locator(".notice")).toContainText("Surface plasmons need TE");
  await expect(page.getByLabel("Experiment")).toHaveValue("");
});

test("silver painted across the stage stays clear of the absorbing border", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await page.getByRole("button", { name: "Silver" }).click();
  const box = (await page.locator(".stage").boundingBox())!;
  const y = box.y + box.height * 0.5;
  await page.mouse.move(box.x + 2, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 2, y, { steps: 20 });
  await page.mouse.up();
  await expect(page.locator(".notice")).toContainText("two cells clear of the absorbing border");
  await expect.poll(async () => JSON.parse(await scene(page)).paint?.runs.length ?? 0).toBeGreaterThan(1);
});

test("the phased array draws its far field over the array factor and steers", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await page.getByLabel("Experiment").selectOption("array");
  await expect(page.getByRole("img", { name: "Far-field pattern" })).toBeVisible();
  await expect(page.locator(".polar-theory")).toHaveCount(1);
  await expect(page.locator(".polar-sim")).toHaveCount(1);
  const phases = async () =>
    (JSON.parse(await scene(page)).sources as { waveform: { phase: number } }[]).map((s) => s.waveform.phase);
  const before = await phases();
  const slider = page.getByLabel("Steer");
  await slider.fill("-40");
  await expect.poll(phases).not.toEqual(before);
  await expect(page.locator(".polar-theory")).toHaveCount(1);
});

test("the double slit shows Fraunhofer until the scene is edited", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await expect(page.locator(".polar-sim")).toHaveCount(1);
  await expect(page.locator(".polar-theory")).toHaveCount(1);
  const box = (await page.locator(".stage").boundingBox())!;
  const x = box.x + box.width * 0.2;
  await page.mouse.move(x, box.y + box.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(x, box.y + box.height * 0.5, { steps: 6 });
  await page.mouse.up();
  await expect(page.locator(".polar-theory")).toHaveCount(0);
});
