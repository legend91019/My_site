import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const pages = ["dist/index.html", "dist/posts/index.html", "dist/search/index.html", "dist/about/index.html"];
const html = await Promise.all(pages.map((page) => readFile(page, "utf8")));
const combined = html.join("\n");

for (const path of ["/posts/", "/search/", "/about/"]) {
  assert.match(combined, new RegExp(`href=\\"${path.replaceAll("/", "\\/")}\\"`), `missing canonical link ${path}`);
  assert.doesNotMatch(combined, new RegExp(`href=\\"${path.slice(0, -1)}\\"`), `found non-canonical link ${path.slice(0, -1)}`);
}

console.log("navigation links use trailing-slash static routes");
