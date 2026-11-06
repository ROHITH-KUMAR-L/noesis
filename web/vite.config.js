import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the FastAPI backend runs on :8000 (python -m memory_rl.web).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
