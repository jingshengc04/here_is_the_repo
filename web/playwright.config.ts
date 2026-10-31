import { defineConfig, devices } from "@playwright/test";

// E2E_BASE_URL=https://<deployment>.vercel.app pnpm e2e   → test a deployment
// pnpm e2e                                              → build + start locally on :3100
const baseURL = process.env.E2E_BASE_URL ?? "http://localhost:3100";
// Protected Vercel URLs: run under `vc env run -- pnpm e2e` so VERCEL_OIDC_TOKEN is injected (never commit it).
const oidc = process.env.E2E_BASE_URL ? process.env.VERCEL_OIDC_TOKEN : undefined;
export const vercelAuthHeaders: Record<string, string> = oidc ? { "x-vercel-trusted-oidc-idp-token": oidc } : {};

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  fullyParallel: true,
  retries: process.env.E2E_BASE_URL ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL,
    viewport: { width: 1280, height: 720 },
    trace: "retain-on-failure",
    extraHTTPHeaders: vercelAuthHeaders,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 720 } } }],
  webServer: [
    ...(process.env.E2E_BASE_URL
      ? []
      : [
          {
            command: "pnpm build && pnpm start -p 3100",
            url: "http://localhost:3100",
            timeout: 180_000,
            reuseExistingServer: true,
            env: { NEXT_PUBLIC_USE_MOCK: "true" },
          },
        ]),
    // Contract fake backend for e2e/live-backend.spec.ts (fast replay).
    { command: "node scripts/fake-backend.ts", url: "http://localhost:8787/api/health", reuseExistingServer: true, env: { PORT: "8787", SPEED: "4" } },
  ],
});
