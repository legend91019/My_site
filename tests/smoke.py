import os

from playwright.sync_api import sync_playwright


def main():
    errors = []
    with sync_playwright() as p:
        browser_options = {"headless": True}
        executable = os.environ.get("BROWSER_EXECUTABLE")
        if executable:
            browser_options["executable_path"] = executable
        browser = p.chromium.launch(**browser_options)
        page = browser.new_page()
        page.on("console", lambda message: errors.append(f"console: {message.text}") if message.type == "error" else None)
        page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))

        base_url = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:4321")
        page.goto(f"{base_url}/", wait_until="networkidle")
        for path, label in [("/posts/", "文章"), ("/search/", "搜索"), ("/about/", "关于")]:
            page.get_by_role("link", name=label, exact=True).click()
            page.wait_for_load_state("networkidle")
            assert page.url.endswith(path), (label, page.url)
            assert page.locator("main").inner_text()
            page.goto(f"{base_url}/", wait_until="networkidle")

        page.goto(f"{base_url}/search/", wait_until="networkidle")
        page.wait_for_selector(".pagefind-ui__search-input", timeout=10000)
        page.locator(".pagefind-ui__search-input").fill("tokenizer")
        page.wait_for_selector(".pagefind-ui__result", timeout=10000)
        assert page.locator(".pagefind-ui__result").count() > 0

        assert not errors, errors
        browser.close()
    print("local navigation and Pagefind smoke test passed")


if __name__ == "__main__":
    main()
