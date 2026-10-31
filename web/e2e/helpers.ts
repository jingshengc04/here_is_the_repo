import { expect, type Page } from "@playwright/test";

export const FAKE_BACKEND = "http://localhost:8787";

export function trackErrors(page: Page) {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

/** Chat through to a complete intake (works with the mock and the fake backend: same scripted agent). */
export async function chatToConfirm(page: Page) {
  const box = page.getByLabel("Message");
  await box.fill("I have about 10k, want growth but can't stomach crypto crashes");
  await box.press("Enter");
  await expect(page.getByTestId("chat-log").locator('[data-role="assistant"]')).toHaveCount(2);
  await box.fill("About 2 years. I'm really into AI infrastructure.");
  await box.press("Enter");
  await expect(page.getByTestId("confirm-card")).toBeVisible();
}

export const openFold = (page: Page, id: string) => page.getByTestId(`fold-${id}`).locator("summary").click();
export const openHolding = (page: Page, symbol: string) => page.getByTestId("holdings-list").locator(`li[data-symbol="${symbol}"] > button`).click();

export async function expectFullReport(page: Page, timeout = 45_000) {
  await expect(page.getByTestId("report")).toBeVisible({ timeout });
  await expect(page.getByTestId("approval")).toHaveAttribute("data-approved", "true");
  await expect(page.getByTestId("rejection-note")).toContainText("first proposal failed");
  await expect(page.getByTestId("donut-3d").locator("[data-symbol]")).toHaveCount(9);
  await expect(page.getByTestId("holdings-list").locator("li[data-symbol]")).toHaveCount(9);
  await expect(page.getByTestId("comparisons")).toContainText("Worst drop");
  await expect(page.getByTestId("backtest-chart").locator(".recharts-area")).toHaveCount(1);
  await expect(page.getByTestId("backtest-chart").locator(".recharts-line")).toHaveCount(1);
  await expect(page.getByTestId("disclaimer")).toContainText("Not financial advice");

  // Details start folded.
  await expect(page.getByTestId("round-1")).toHaveCount(0);
  await expect(page.getByTestId("evidence")).toHaveCount(0);

  await page.getByRole("button", { name: "Compare more benchmarks" }).click();
  await expect(page.getByTestId("backtest-chart").locator(".recharts-line")).toHaveCount(3);

  await page.getByRole("button", { name: "See what changed" }).click();
  await expect(page.getByTestId("round-1")).toContainText("Rejected");
  await expect(page.getByTestId("round-diff")).toContainText("LLY 7%");
  // Backtest.sharpeCi: round 1 CI crosses 0 (red), final round entirely above 0 (green).
  await expect(page.getByTestId("round-1").getByTestId("sharpe-ci")).toHaveAttribute("data-crosses-zero", "true");
  await expect(page.getByTestId("final-round").getByTestId("sharpe-ci")).toHaveAttribute("data-crosses-zero", "false");
  await expect(page.getByTestId("round-1")).toContainText("Sharpe 95% CI lower bound > 0");

  await openFold(page, "metrics");
  await expect(page.getByTestId("verified-badge")).toHaveAttribute("data-match", "true");
  await expect(page.getByTestId("sharpe-ci-text")).toContainText("Sharpe 0.74 (95% CI 0.07–1.41)");
  await expect(page.getByTestId("metrics-table").locator("tbody tr")).toHaveCount(6);
}
