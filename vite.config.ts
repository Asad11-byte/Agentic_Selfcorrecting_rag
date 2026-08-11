import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // FastAPI runs separately (uvicorn) on port 8000 — this proxies /api
      // calls from `npm run dev` straight through to it.
      "/api": "http://localhost:8000"
    }
  }
});
