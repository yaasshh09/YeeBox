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
