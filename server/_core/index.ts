import "dotenv/config";
import express from "express";
import { createServer } from "http";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { registerOAuthRoutes } from "./oauth";
import { registerStorageProxy } from "./storageProxy";
import { appRouter } from "../routers";
import { createContext } from "./context";
import { registerInferenceProxy } from "../inferenceProxy";
import { registerExplainRoute } from "../generateExplanation";
import { registerEventSummaryRoute } from "../eventSummary";
import { serveStatic, setupVite } from "./vite";

async function startServer() {
  const app = express();
  const server = createServer(app);
  // Base64 image bodies fit within the JSON parser's request limit.
  app.use(express.json({ limit: "70mb" }));
  app.use(express.urlencoded({ limit: "50mb", extended: true }));
  registerStorageProxy(app);
  registerOAuthRoutes(app);
  registerInferenceProxy(app);
  registerExplainRoute(app);
  registerEventSummaryRoute(app);
  app.use(
    "/api/trpc",
    createExpressMiddleware({
      router: appRouter,
      createContext,
    })
  );

  if (process.env.NODE_ENV === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  const port = Number(process.env.PORT || "3000");
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error(`Invalid PORT value: ${process.env.PORT}`);
  }

  server.listen(port, "0.0.0.0", () => {
    console.log(`Baby Action AI listening on port ${port}`);
  });
  server.on("error", (error) => {
    console.error(`Could not start the website on port ${port}`, error);
    process.exitCode = 1;
  });
}

startServer().catch((error) => {
  console.error("Could not initialize Baby Action AI", error);
  process.exitCode = 1;
});
