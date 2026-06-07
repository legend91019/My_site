import { cpSync, existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const sourceDir = resolve("dist/client/pagefind");
const vercelStaticDir = resolve(".vercel/output/static/pagefind");

if (!existsSync(sourceDir)) {
  process.exit(0);
}

mkdirSync(vercelStaticDir, { recursive: true });
cpSync(sourceDir, vercelStaticDir, { recursive: true, force: true });
