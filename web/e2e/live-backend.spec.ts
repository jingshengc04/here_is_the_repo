import { expect, test } from "@playwright/test";
import { chatToConfirm, expectFullReport, FAKE_BACKEND, openFold, openHolding, trackErrors } from "./helpers";

// Real network path (fetch + EventSource + CORS) against scripts/fake-backend.ts.
// The backend allows one run at a time, so these run serially.
test.describe.configure({ mode: "serial" });
test.skip(!!process.env.E2E_BASE_URL, "fake backend is local-only; a deployed https page can't reach localhost");

const LIVE = `/?mock=0&backend=${encodeURIComponent(FAKE_BACKEND)}`;

async function waitIdle(request: import("@playwright/test").APIRequestContext) {
  // Let any previous run finish so the next POST /api/runs isn't 429.
  for (let i = 0; i < 40; i++) {
    const res = await request.post(`${FAKE_BACKEND}/api/runs`, { data: { amountUsd: 100, horizonYears: 1, riskTolerance: "MODERATE" } });
    if (res.status() === 202) {
      const { runId } = await res.json();
      await request.get(`${FAKE_BACKEND}/api/runs/${runId}/events`); // streams to the end
      return;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
}

test.beforeEach(async ({ request }) => waitIdle(request));

test("live SSE: chat via backend → run → report", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto(LIVE);
  await expect(page.getByTestId("mode-badge")).toContainText(FAKE_BACKEND);
  await chatToConfirm(page);
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("run-status")).toContainText("Live");
  await expect(page.getByTestId("agent-VERIFIER").first()).toHaveAttribute("data-state", "rejected", { timeout: 20_000 });
  await expectFullReport(page);
  expect(errors, errors.join("\n")).toEqual([]);
});

test("SSE blocked → falls back to polling the snapshot", async ({ page }) => {
  await page.route("**/events", (r) => r.abort());
  await page.goto(LIVE);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("25000");
  await page.getByRole("button", { name: "5 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("run-status")).toContainText("polling every 3s");
  await expectFullReport(page);
  await expect(page.getByTestId("holdings-list").locator('li[data-symbol="NVDA"]')).toContainText("$3,500"); // 14% of $25,000
});

test("agent.failed marks the lane and the run still completes", async ({ page }) => {
  await page.route("**/api/runs", (r) =>
    r.request().method() === "POST" ? r.continue({ url: `${FAKE_BACKEND}/api/runs?scenario=agent_failed` }) : r.continue(),
  );
  await page.goto(LIVE);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "2 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("agent-SOCIAL")).toHaveAttribute("data-state", "failed", { timeout: 15_000 });
  await expect(page.getByTestId("agent-SOCIAL")).toContainText("Search quota exceeded");
  await expect(page.getByTestId("report")).toBeVisible({ timeout: 30_000 });
});

test("run.failed shows the error and a retry", async ({ page }) => {
  await page.route("**/api/runs", (r) =>
    r.request().method() === "POST" ? r.continue({ url: `${FAKE_BACKEND}/api/runs?scenario=run_failed` }) : r.continue(),
  );
  await page.goto(LIVE);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "2 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("run-failed")).toContainText("Gemini sandbox timed out", { timeout: 20_000 });
  // Try again re-sends the same intake...
  const retry = page.waitForRequest((r) => r.method() === "POST" && r.url().includes("/api/runs"));
  await page.getByRole("button", { name: "Try again" }).click();
  expect((await retry).postDataJSON()).toMatchObject({ amountUsd: 10000, horizonYears: 2, riskTolerance: "MODERATE" });
  await expect(page.getByTestId("run-failed")).toContainText("Gemini sandbox timed out", { timeout: 20_000 });
  // ...and Edit and run again returns to the filled-in form.
  await page.getByRole("button", { name: "Edit and run again" }).click();
  await expect(page.getByTestId("confirm-card")).toBeVisible();
  await expect(page.getByLabel("Amount (USD)")).toHaveValue("10000");
});

test("429 while another run is in progress → waits 5 s and retries until it starts", async ({ page, request }) => {
  const busy = await request.post(`${FAKE_BACKEND}/api/runs`, { data: { amountUsd: 500, horizonYears: 1, riskTolerance: "MODERATE" } });
  expect(busy.status()).toBe(202);
  await page.goto(LIVE);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "2 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByText("Another run is in progress. Retrying in 5 seconds")).toBeVisible();
  await expect(page.getByTestId("confirm-card")).toBeVisible();
  // The busy run finishes (~6 s at SPEED=4); a retry then gets through.
  await expect(page.getByTestId("run-status")).toBeVisible({ timeout: 30_000 });
});

test("numbersVerified=false shows the ⚠ badge on the finding", async ({ page }) => {
  await page.route("**/api/runs", (r) =>
    r.request().method() === "POST" ? r.continue({ url: `${FAKE_BACKEND}/api/runs?scenario=unverified_number` }) : r.continue(),
  );
  await page.goto(LIVE);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "2 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("findings-feed").getByTestId("numbers-unverified")).toHaveCount(1, { timeout: 15_000 });
  await expect(page.getByTestId("report")).toBeVisible({ timeout: 30_000 });
  await openHolding(page, "NVDA");
  await expect(page.getByTestId("evidence").getByTestId("numbers-unverified")).toContainText("don't match data");
});

test("?replay=<name> is forwarded to POST /api/runs; unknown replay shows the 404 message", async ({ page }) => {
  await page.goto(`${LIVE}&replay=demo`);
  await expect(page.getByTestId("mode-badge")).toContainText("replay demo");
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "2 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  const post = page.waitForRequest((r) => r.method() === "POST" && r.url().includes("/api/runs"));
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  expect((await post).url()).toContain("/api/runs?replay=demo");
  await expect(page.getByTestId("report")).toBeVisible({ timeout: 30_000 });

  await page.goto(`${LIVE}&replay=nope`);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "2 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByText("Replay 'nope' not found")).toBeVisible();
});

test("sandbox unavailable → 'Not cross-checked', not 'Numbers differ'", async ({ page }) => {
  await page.route("**/api/runs", (r) =>
    r.request().method() === "POST" ? r.continue({ url: `${FAKE_BACKEND}/api/runs?scenario=no_sandbox` }) : r.continue(),
  );
  await page.goto(LIVE);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "2 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("report")).toBeVisible({ timeout: 30_000 });
  await openFold(page, "metrics");
  await expect(page.getByTestId("verified-badge")).toHaveAttribute("data-state", "unchecked");
  await expect(page.getByTestId("verified-badge")).toContainText("Not cross-checked");
});

test("personalised report: BONDS slice, 'You asked for this' tag, preference notes", async ({ page }) => {
  await page.route("**/api/runs", (r) =>
    r.request().method() === "POST" ? r.continue({ url: `${FAKE_BACKEND}/api/runs?scenario=personalised` }) : r.continue(),
  );
  await page.goto(LIVE);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "3 years" }).click();
  await page.getByRole("button", { name: "Conservative" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("report")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("holdings-list").getByRole("region", { name: "Bonds" })).toContainText("BND");
  await expect(page.getByTestId("donut-3d").locator('[data-symbol="BND"]')).toHaveCount(1);
  await expect(page.getByTestId("holdings-list").locator('li[data-symbol="NVDA"]').getByTestId("user-requested")).toBeVisible();
  await expect(page.getByTestId("preference-notes")).toContainText("excluded industry Tobacco");
  await page.getByRole("button", { name: "Edit and run again" }).click();
  await expect(page.getByRole("button", { name: "3 years" })).toHaveAttribute("aria-pressed", "true");
});

test("confirm card stays once shown when the chat asks an optional follow-up", async ({ page }) => {
  let turn = 0;
  await page.route("**/api/intake/messages", (r) => {
    turn += 1;
    const draft = { amountUsd: 10000, horizonYears: 2, riskTolerance: "MODERATE", theme: null, preferences: { exclude: ["tobacco"] } };
    const body =
      turn === 1
        ? { reply: "All set. Saved: no tobacco", intake: draft, isComplete: true }
        : { reply: "Should bonds still be included?", intake: { ...draft, horizonYears: null }, isComplete: false };
    return r.fulfill({ json: body, headers: { "Access-Control-Allow-Origin": "*" } });
  });
  await page.goto(LIVE);
  const box = page.getByLabel("Message");
  await box.fill("10k, 2 years, moderate, no tobacco");
  await box.press("Enter");
  await expect(page.getByTestId("confirm-card")).toBeVisible();
  await expect(page.getByTestId("saved-line")).toContainText("no tobacco");
  await box.fill("hmm, what about bonds?");
  await box.press("Enter");
  await expect(page.getByTestId("chat-log")).toContainText("Should bonds still be included?");
  await expect(page.getByTestId("confirm-card")).toBeVisible();
  await expect(page.getByRole("button", { name: "2 years" })).toHaveAttribute("aria-pressed", "true"); // null didn't wipe it
  await expect(page.getByRole("button", { name: /build my portfolio/ })).toBeEnabled();
});

test("not approved: badge, and notes headed with why", async ({ page }) => {
  await page.route("**/api/runs", (r) =>
    r.request().method() === "POST" ? r.continue({ url: `${FAKE_BACKEND}/api/runs?scenario=not_approved` }) : r.continue(),
  );
  await page.goto(LIVE);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("10000");
  await page.getByRole("button", { name: "2 years" }).click();
  await page.getByRole("button", { name: "Moderate" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("approval")).toHaveAttribute("data-approved", "false", { timeout: 30_000 });
  await expect(page.getByTestId("preference-notes")).toContainText("Why it wasn't approved");
  await expect(page.getByTestId("preference-notes")).toContainText("beyond the 35%");
});
