import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

const proxy = { "/api": process.env.API_URL ?? "http://localhost:8000" };

// `vite build --mode static` produces one self-contained index.html (JS, CSS and fonts inlined);
// scripts/export_static.py then embeds a worklist snapshot so it opens straight from disk.
export default defineConfig(({ mode }) => ({
  plugins: mode === "static" ? [react(), viteSingleFile()] : [react()],
  server: { proxy },
  preview: { proxy },
  build: mode === "static" ? { outDir: "dist-static", assetsInlineLimit: 100_000_000 } : {},
}));
