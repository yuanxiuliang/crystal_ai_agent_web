import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./web",
  testMatch: "**/*.spec.ts",
  timeout: 120_000,
  forbidOnly: Boolean(process.env.CI),
  fullyParallel: false,
  retries: 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:3003",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
});
