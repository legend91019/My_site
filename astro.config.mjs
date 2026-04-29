import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import keystatic from "@keystatic/astro";
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  // 核心修复点：请确保这里是一个纯净的字符串，结尾不要带斜杠，前后不要有空格
  site: "https://my-site-ebon-eta.vercel.app", 
  
  integrations: [
    mdx(), 
    sitemap(), 
    react(), 
    keystatic()
  ],

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: {
        light: "github-light",
        dark: "github-dark",
      },
      wrap: true,
    },
  },
});
