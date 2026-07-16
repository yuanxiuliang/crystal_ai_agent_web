import { expect, test, type Page } from "@playwright/test";

const password = "e2e-password-12345";
const workflowAnswerTimeout = 90_000;

function email(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}@e2e.invalid`;
}

async function login(page: Page, userEmail: string, value = password): Promise<void> {
  await page.goto("/login");
  await page.locator('input[name="email"]').fill(userEmail);
  await page.locator('input[name="password"]').fill(value);
  await page.getByRole("button", { name: "继续" }).click();
}

async function ask(page: Page, question: string): Promise<void> {
  const composer = page.getByRole("textbox", { name: "研究问题" });
  await composer.fill(question);
  await page.getByTitle("发送").click();
}

test("a user can register, edit a real-record statistic, and view an explicitly unverified prediction", async ({ page }) => {
  // The API E2E contract exercises successful real-LLM generation. This browser journey targets
  // deterministic aggregate retrieval, edit/regeneration, and local prediction rendering.
  test.setTimeout(4 * 60_000);
  const userEmail = email("browser");
  await login(page, userEmail);
  await expect(page).toHaveURL(/\/chat/);
  await expect(page.getByRole("textbox", { name: "研究问题" })).toBeVisible();

  await ask(page, "Eu基化合物一般采用哪些单晶生长方法？");
  const aggregateAnswer = page.locator(".chat-message.assistant").last();
  await expect(aggregateAnswer.getByText("真实记录统计")).toBeVisible({
    timeout: workflowAnswerTimeout,
  });
  await expect(aggregateAnswer).toContainText("方法分布");
  await expect(aggregateAnswer).toContainText("EuCr2As2");
  const literatureTable = aggregateAnswer.getByTestId("literature-record-table");
  await expect(literatureTable).toBeVisible();
  await expect(literatureTable).toContainText("flux growth");
  await expect(literatureTable).toContainText("10.5555/e2e.eucr2as2.001");
  await expect(aggregateAnswer.getByText(/条文献证据/)).toBeVisible();
  await aggregateAnswer.getByText(/条文献证据/).click();
  await expect(page.getByLabel("证据来源")).toContainText("10.5555/e2e.eucr2as2.001");
  await page.getByTitle("关闭证据面板").click();

  const initialQuestion = page.locator(".chat-message.user").first();
  await initialQuestion.getByTitle("编辑问题").click();
  await initialQuestion
    .getByRole("textbox", { name: "编辑问题" })
    .fill("碘传输剂使用哪些化合物的单晶生长呢？");
  await initialQuestion.getByTitle("更新并重新生成").click();
  const editedAnswer = page.locator(".chat-message.assistant").last();
  await expect(editedAnswer.getByText("真实记录统计")).toBeVisible({
    timeout: workflowAnswerTimeout,
  });
  await expect(initialQuestion).toContainText("碘传输剂使用哪些化合物的单晶生长呢？");
  await expect(page.locator(".chat-message.user")).toHaveCount(1);
  await expect(editedAnswer).toContainText("TaAs");
  await expect(editedAnswer).toContainText("ZnIn2S4");
  await expect(editedAnswer.getByTestId("literature-record-table")).toBeVisible();

  await page.getByRole("button", { name: "新建对话" }).click();
  await expect(page.getByRole("textbox", { name: "研究问题" })).toBeVisible();
  await ask(page, "我要做 Mn3ZnN");
  const predictionAnswer = page.locator(".chat-message.assistant").last();
  await expect(predictionAnswer.getByText("可尝试方案 · 模型预测 · 未验证")).toBeVisible({
    timeout: workflowAnswerTimeout,
  });
  await expect(predictionAnswer).toContainText("Mn3ZnN");
  await expect(predictionAnswer.locator(".assistant-content")).toBeVisible();
  await expect(predictionAnswer.locator(".prediction-content")).toBeVisible();
  await expect(predictionAnswer.getByTestId("prediction-route-table")).toBeVisible();
  await expect(predictionAnswer).toContainText("不是文献事实");
  await expect(predictionAnswer.getByText("候选路线与限制")).toBeVisible();
});

test("an existing email rejects a wrong password before allowing the correct password", async ({ page }) => {
  const userEmail = email("login");
  await login(page, userEmail);
  await expect(page).toHaveURL(/\/chat/);
  await page.getByRole("button", { name: "退出登录" }).click();
  await expect(page).toHaveURL(/\/login/);

  await login(page, userEmail, "wrong-password-123");
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page).toHaveURL(/\/login/);

  await login(page, userEmail);
  await expect(page).toHaveURL(/\/chat/);
});
