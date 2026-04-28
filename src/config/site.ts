export const siteConfig = {
  title: "天庭",
  description: "记录王勇顺的AI学习,工程实践,以及各种经历。",
  siteUrl: "https://example.com",
  author: "王勇顺",
  github: "https://github.com/legend91019/My_site",
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
