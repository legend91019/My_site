export const siteConfig = {
  title: "林风技术博客",
  description: "记录 NLP、LLM、机器学习与工程实践的技术长文。",
  siteUrl: "https://example.com",
  author: "Your Name",
  github: "https://github.com/yourname",
  nav: [
    { href: "/", label: "首页" },
    { href: "/posts", label: "文章" },
    { href: "/search", label: "搜索" },
    { href: "/about", label: "关于" },
  ],
  seo: {
    defaultOgImage: "/og-default.png",
    twitter: "@yourhandle",
  },
  comments: {
    provider: "giscus",
    repo: "yourname/yourrepo",
    repoId: "R_kgDOExample",
    category: "General",
    categoryId: "DIC_kwDOExample4CExample",
  },
} as const;
