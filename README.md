# 现代技术博客（Astro）

基于 Astro + Markdown/MDX 的现代技术博客，支持：

- 文章、分类、标签
- 站内搜索（Pagefind）
- 评论系统（giscus）
- 深浅色主题切换
- RSS + Sitemap + SEO 元信息

## 本地开发

```bash
npm install
npm run dev
```

## 构建

```bash
npm run build
npm run preview
```

构建时会自动生成 `dist/pagefind` 搜索索引。

## 内容写作

文章放在 `src/content/posts/`，Frontmatter 格式：

```md
---
title: "文章标题"
summary: "文章摘要"
date: 2026-04-20
updated: 2026-04-24
tags: ["标签1", "标签2"]
category: "分类名"
draft: false
---
```

## 评论配置

在 `src/config/site.ts` 修改 giscus 参数：

- `comments.repo`
- `comments.repoId`
- `comments.category`
- `comments.categoryId`

## 部署建议

- 平台：Vercel 或 Netlify
- 构建命令：`npm run build`
- 输出目录：`dist`
