import { getCollection } from "astro:content";
import rss from "@astrojs/rss";
import { siteConfig } from "../config/site";
import { sortPosts } from "../utils/posts";

export async function GET(context) {
  const posts = sortPosts(await getCollection("posts"));
  return rss({
    title: siteConfig.title,
    description: siteConfig.description,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.summary,
      link: `/posts/${post.id}/`,
    })),
  });
}
