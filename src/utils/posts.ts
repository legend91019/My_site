import type { CollectionEntry } from "astro:content";

export function sortPosts(posts: CollectionEntry<"posts">[]) {
  return posts
    .filter((post) => !post.data.draft)
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export function readingTimeFromText(text: string) {
  const wordsPerMinute = 260;
  const words = text.trim().split(/\s+/u).filter(Boolean).length;
  return Math.max(1, Math.ceil(words / wordsPerMinute));
}
