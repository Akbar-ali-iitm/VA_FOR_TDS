import os
import json
import re
from datetime import datetime
from urllib.parse import urljoin
from markdownify import markdownify as md
from playwright.sync_api import sync_playwright
from pathlib import PurePosixPath

# CONFIGURATION
BASE_URL = "https://tds.s-anand.net/#/2025-01/"
OUTPUT_DIR = "tds_markdown"
METADATA_FILE = "metadata.json"

# State
visited = set()
metadata = []

def sanitize_filename(title):
    """Remove invalid filename characters."""
    return re.sub(r'[\/*?:"<>|]', "_", title).strip().replace(" ", "_")

def extract_sidebar_links(page):
    """Get unique hrefs from sidebar."""
    return list(set(
        page.eval_on_selector_all(".sidebar-nav a[href]", "els => els.map(el => el.href)")
    ))

def wait_for_main_article(page):
    """Wait and return main article HTML."""
    page.wait_for_selector("article.markdown-section#main", timeout=10000)
    return page.inner_html("article.markdown-section#main")


def crawl_page(page, url):
    if url in visited:
        return
    visited.add(url)

    print(f"📄 Visiting: {url}")
    try:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(300)  # wait for JS to finish loading
        html = wait_for_main_article(page)
    except Exception as e:
        print(f"❌ Failed: {url} — {e}")
        return

    # Extract slug from #/ path (e.g., 'data-sourcing/scraping-imdb-with-javascript.md')
    raw_path = url.split("#/")[-1].strip("/")
    filename = PurePosixPath(raw_path).name  # gets just 'scraping-imdb-with-javascript.md'

    # Ensure .md extension
    if not filename.endswith(".md"):
        filename += ".md"

    # Clean and sanitize filename
    filename = filename.lower().replace(" ", "-")
    filename = re.sub(r"[^\w\-.]", "_", filename)

    filepath = os.path.join(OUTPUT_DIR, filename)

    # Convert HTML to Markdown
    markdown = md(html).strip()

    # Save the Markdown file with frontmatter
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f'title: "{page.title().split(" - ")[0].strip()}"\n')
        f.write(f'original_url: "{url}"\n')
        f.write(f'downloaded_at: "{datetime.now().isoformat()}"\n')
        f.write("---\n\n")
        f.write(markdown)

    # Save metadata for reference
    metadata.append({
        "title": page.title().split(" - ")[0].strip(),
        "filename": filename,
        "original_url": url,
        "downloaded_at": datetime.now().isoformat()
    })



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("🌍 Launching browser...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print("📦 Loading main page...")
        page.goto(BASE_URL, wait_until="domcontentloaded")
        
        # ✅ Wait for sidebar to fully load
        try:
            page.wait_for_selector(".sidebar-nav a[href]", timeout=10000)
        except Exception:
            print("❌ Sidebar failed to load.")
            return

        page.wait_for_timeout(1000)  # buffer wait

        links = extract_sidebar_links(page)
        print(f"🔗 Found {len(links)} sidebar links.")

        for link in links:
            crawl_page(page, link)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Done! {len(metadata)} files saved.")
        browser.close()


if __name__ == "__main__":
    main()
