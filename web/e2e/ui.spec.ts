import { expect, test } from "@playwright/test";
import { chatToConfirm, expectFullReport, openFold, openHolding, trackErrors } from "./helpers";

// Mock mode (fixture replay in the browser, no backend).

test("chat → confirm → agents stream → round-1 rejection → approved report", async ({ page }) => {
  test.setTimeout(90_000);
  const errors = trackErrors(page);
  await page.goto("/?mock=1");
  await expect(page.getByTestId("chat-log")).toContainText("How much are you looking to invest");

  await chatToConfirm(page);
  await expect(page.getByTestId("intake-draft")).toContainText("$10,000");
  await expect(page.getByLabel("Amount (USD)")).toHaveValue("10000");
  await expect(page.getByRole("button", { name: "2 years" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "Moderate" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByLabel(/Theme/)).toHaveValue("AI infrastructure");

  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page).toHaveURL(/run=mock_/);
  await expect(page.getByTestId("agent-DISCOVERY")).toHaveAttribute("data-state", "running");
  await expect(page.getByTestId("findings-feed")).toContainText("DISCOVERY", { timeout: 10_000 });
  // The demo moment: Verifier rejects round 1 while round 2 is still pending.
  await expect(page.getByTestId("agent-VERIFIER").first()).toHaveAttribute("data-state", "rejected", { timeout: 30_000 });
  await expect(page.getByTestId("agent-VERIFIER").first()).toContainText("Changes requested");

  await expectFullReport(page);
  // Evidence opens inline; NVDA's FINANCIALS claim has numbers → audited ✓.
  await openHolding(page, "NVDA");
  await expect(page.getByTestId("evidence").getByTestId("numbers-verified")).toHaveCount(1);
  await openHolding(page, "LLY");
  await expect(page.getByTestId("evidence")).toHaveCount(1);
  await expect(page.getByTestId("evidence")).toContainText("Landscape");
  await expect(page.getByTestId("evidence")).toContainText("Case for");

  await page.screenshot({ path: "test-results/report-1280x720.png" });
  await openFold(page, "activity");
  await expect(page.getByTestId("agent-lanes")).toBeVisible();
  await openFold(page, "findings");
  await expect(page.getByTestId("fold-findings")).toContainText("Discovery");
  expect(errors, errors.join("\n")).toEqual([]);
});

test("skip chat → plain form; out-of-range amount shows the 422 message inline", async ({ page }) => {
  await page.goto("/?mock=1");
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("50");
  await page.getByRole("button", { name: "3 years" }).click();
  await page.getByRole("button", { name: "Aggressive" }).click();
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Must be between 100 and 10,000,000" })).toBeVisible();
  await expect(page.getByLabel("Amount (USD)")).toHaveAttribute("aria-invalid", "true");
});

test("months horizon and preferences flow from chat → confirm chips → request → report notes", async ({ page }) => {
  await page.goto("/?mock=1");
  const box = page.getByLabel("Message");
  await box.fill("I have 20k, moderate risk, for 18 months. No tobacco, and include NVDA");
  await box.press("Enter");
  await expect(page.getByTestId("confirm-card")).toBeVisible();
  await expect(page.getByTestId("intake-draft")).toContainText("18 months");
  await expect(page.getByTestId("intake-draft")).toContainText("no tobacco");
  await expect(page.getByTestId("saved-line").last()).toContainText("no tobacco");
  await expect(page.getByTestId("saved-line").last()).toContainText("include NVDA");
  await expect(page.getByTestId("preferences")).toHaveAttribute("open", "");
  await expect(page.getByRole("button", { name: "Remove tobacco" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Remove NVDA" })).toBeVisible();
  // Edit a chip and a style before running.
  await page.getByLabel("Exclude", { exact: true }).fill("oil");
  await page.getByLabel("Exclude", { exact: true }).press("Enter");
  await page.getByRole("button", { name: "Dividend" }).click();
  await page.getByLabel("Horizon in years").fill("0.5");
  await expect(page.getByTestId("confirm-card")).toContainText("6 months");
  await page.getByRole("button", { name: /build my portfolio/ }).click();
  await expect(page.getByTestId("report")).toBeVisible({ timeout: 45_000 });
  await expect(page.getByTestId("preference-notes")).toContainText('"oil" excluded');
  await expect(page.getByTestId("preference-notes")).toContainText("Dividend style");
  await expect(page.getByTestId("report")).toContainText("Planned for 6 months");
});

test("target mix over 100% blocks Run", async ({ page }) => {
  await page.goto("/?mock=1");
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await page.getByLabel("Amount (USD)").fill("5000");
  await page.getByRole("button", { name: "1 year" }).click();
  await page.getByRole("button", { name: "Aggressive" }).click();
  await page.getByText("Add personal preferences").click();
  await page.getByLabel("Crypto target %").fill("70");
  await page.getByLabel("Stocks target %").fill("40");
  await expect(page.getByText("Targets add up to 110%")).toBeVisible();
  await expect(page.getByRole("button", { name: /build my portfolio/ })).toBeDisabled();
});

test("form → Back to chat keeps the conversation", async ({ page }) => {
  await page.goto("/?mock=1");
  const box = page.getByLabel("Message");
  await box.fill("I have about 10k");
  await box.press("Enter");
  await expect(page.getByTestId("chat-log").locator('[data-role="assistant"]')).toHaveCount(2);
  await page.getByRole("button", { name: "Prefer a form? Skip the chat" }).click();
  await expect(page.getByLabel("Amount (USD)")).toHaveValue("10000"); // draft carried into the form
  await page.getByRole("button", { name: "← Back to chat" }).click();
  await expect(page.getByTestId("chat-log")).toContainText("I have about 10k");
  await expect(page.getByLabel("Message")).toBeVisible();
});

test("deep link ?run=<id> restores a finished report", async ({ page }) => {
  await page.goto("/?mock=1&run=run_fixture");
  await expectFullReport(page, 10_000);
  await openHolding(page, "NVDA");
  await expect(page.getByTestId("evidence").getByTestId("numbers-verified")).toHaveCount(1);
});
