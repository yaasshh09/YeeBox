import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  // Relative asset paths, so the same build serves from the GitHub Pages
  // subpath and from a local preview.
  base: "./",
  plugins: [react()],
  worker: { format: "es" },
  server: {
    // The presets and golden files live in the repo's validation folder.
    fs: { allow: [".."] },
  },
  test: {
    // The uppercase test reads index.css as text; without this Vitest stubs
    // CSS to an empty string and the test would pass vacuously.
    css: true,
  },
});
