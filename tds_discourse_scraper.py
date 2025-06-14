import os
import re
import json
import requests
import mimetypes
from tqdm import tqdm
from datetime import datetime
from bs4 import BeautifulSoup
import google.generativeai as genai
from playwright.sync_api import sync_playwright, TimeoutError

# --- Configuration ---
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34
CATEGORY_JSON_URL = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json"
AUTH_STATE_FILE = "auth.json"
API_KEY = os.getenv("GENAI_API_KEY")
MODEL = "gemini-2.0-flash"
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14)
OUTPUT_JSON = "tds_discourse_json"
OUTPUT_MARKDOWN = "tds_discourse_md"

# --- Setup ---
os.makedirs(OUTPUT_JSON, exist_ok=True)
os.makedirs(OUTPUT_MARKDOWN, exist_ok=True)
genai.configure(api_key=API_KEY)
client = genai.GenerativeModel(MODEL)

# --- Gemini Image Understanding ---
def describe_image(image_source):
    try:
        if image_source.startswith("http"):
            image_bytes = requests.get(image_source).content
            mime_type = mimetypes.guess_type(image_source)[0] or "image/jpeg"
        else:
            with open(image_source, "rb") as f:
                image_bytes = f.read()
            mime_type = mimetypes.guess_type(image_source)[0] or "image/jpeg"

        response = client.generate_content(
            contents=[
                {"mime_type": mime_type, "data": image_bytes},
                "Understand and explain this image in the context of data science or programming."
            ]
        )
        return response.text.strip()
    except Exception as e:
        return f"[Image failed: {e}]"

# --- Convert HTML to Markdown with image understanding ---
def convert_html_to_markdown(html: str, base_url: str = "", local_img_dir: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src:
            continue
        src_full = src if src.startswith("http") else os.path.join(local_img_dir if not src.startswith("/") else base_url, src)
        desc = describe_image(src_full)
        img.replace_with(desc)
    return soup.get_text(separator="\n").strip()

# --- Write Markdown file ---
def save_markdown(content: str, title: str, url: str, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"title: {title}\n")
        f.write(f"original_url: {url}\n")
        f.write(f"downloaded_at: {datetime.now().isoformat()}\n")
        f.write("---\n\n")
        f.write(content)

# --- Parse Discourse date ---
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

# --- Playwright-based scraping ---
def login_and_save_auth(playwright):
    print("🔐 No auth found. Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{BASE_URL}/login")
    print("🌐 Please log in manually using Google. Then press ▶️ (Resume) in Playwright bar.")
    page.pause()
    context.storage_state(path=AUTH_STATE_FILE)
    print("✅ Login state saved.")
    browser.close()

# --- Auth check ---
def is_authenticated(page):
    try:
        page.goto(CATEGORY_JSON_URL, timeout=10000)
        page.wait_for_selector("pre", timeout=5000)
        json.loads(page.inner_text("pre"))
        return True
    except (TimeoutError, json.JSONDecodeError):
        return False

# --- Main scrape logic ---
def scrape_posts(playwright):
    print("🔍 Starting scrape using saved session...")
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(storage_state=AUTH_STATE_FILE)
    page = context.new_page()

    page_num = 0
    found_in_range = 0

    while True:
        paginated_url = f"{CATEGORY_JSON_URL}?page={page_num}"
        print(f"📦 Fetching page {page_num}...")
        page.goto(paginated_url)
        try:
            data = json.loads(page.inner_text("pre"))
        except:
            data = json.loads(page.content())
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break

        for topic in tqdm(topics):
            created_at = parse_date(topic["created_at"])
            if not (START_DATE <= created_at <= END_DATE):
                continue

            topic_url = f"{BASE_URL}/t/{topic['slug']}/{topic['id']}.json"
            page.goto(topic_url)
            try:
                topic_data = json.loads(page.inner_text("pre"))
            except:
                topic_data = json.loads(page.content())

            found_in_range += 1

            json_filename = f"{topic['id']}_{topic['slug']}.json"
            with open(os.path.join(OUTPUT_JSON, json_filename), "w", encoding="utf-8") as f:
                json.dump(topic_data, f, indent=2)

            md_filename = f"{topic['id']}_{topic['slug']}.md"
            md_path = os.path.join(OUTPUT_MARKDOWN, md_filename)
            title = topic_data["title"].strip().replace("\n", "")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")
                f.write(f"*Original URL: {BASE_URL}/t/{topic['slug']}/{topic['id']}*\n")
                f.write(f"*Posted on: {created_at.isoformat()}*\n\n")
                for post in topic_data["post_stream"]["posts"]:
                    username = post["username"]
                    posted_at = datetime.strptime(post["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M")
                    content = convert_html_to_markdown(post["cooked"], base_url=BASE_URL)
                    f.write(f"---\n**{username}** posted on {posted_at}:\n\n{content}\n\n")

        page_num += 1

    print(f"✅ Completed. Total topics in date range: {found_in_range}")
    browser.close()

# --- Main Entrypoint ---
def main():
    with sync_playwright() as p:
        if not os.path.exists(AUTH_STATE_FILE):
            login_and_save_auth(p)
        else:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(storage_state=AUTH_STATE_FILE)
            page = context.new_page()
            if not is_authenticated(page):
                print("⚠️ Session invalid. Re-authenticating...")
                browser.close()
                login_and_save_auth(p)
            else:
                print("✅ Using existing authenticated session.")
                browser.close()
        scrape_posts(p)

if __name__ == "__main__":
    main()
