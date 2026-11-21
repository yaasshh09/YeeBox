import { defineConfig } from "@playwright/test";

// Runs against the production build. Headless Chromium has no WebGPU
// adapter, so these cover the CPU fallback path; the GPU self-test page is
// checked locally in a headed browser (see PROGRESS.md).
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  use: { baseURL: "http://localhost:4173", viewport: { width: 1280, height: 800 } },
  webServer: {
    command: "npx vite preview --port 4173 --strictPort",
    port: 4173,
    reuseExistingServer: !process.env.CI,
  },
});
