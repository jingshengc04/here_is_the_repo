// Runtime config. URL params override the build-time env so one deployment can be pointed at any backend:
//   ?mock=0&backend=https://<tunnel-or-cloud-run>   → live backend
//   ?mock=1                                          → fixture replay, backend not needed
const param = (k: string) => (typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get(k));

export function backendUrl(): string {
  return (param("backend") ?? process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000").replace(/\/+$/, "");
}

/** ?replay=<name> → POST /api/runs?replay=<name>: the backend replays a saved run (stage safety net). */
export function replayName(): string | null {
  const p = param("replay");
  return p && /^[A-Za-z0-9_-]+$/.test(p) ? p : null;
}

export function isMock(): boolean {
  const p = param("mock");
  if (p !== null) return p === "1" || p === "true";
  return process.env.NEXT_PUBLIC_USE_MOCK === "true";
}
