import os
import re
import time
import json
import numpy as np
import tiktoken
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

OPENAI_API_KEY = os.getenv("API_KEY")
MODEL = "text-embedding-3-small"
CHUNK_SIZE = 200
OVERLAP = 50
SUBCHUNK_TOKENS = 150
OUTPUT_FILE = "embeddings.npz"

# --- Setup ---
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://aipipe.org/openai/v1")
enc = tiktoken.encoding_for_model(MODEL)

# --- Utilities ---
def slugify(name):
    return name.lower().replace("_", "-").replace(" ", "-")

def extract_slug_id(filename):
    match = re.match(r"(\d+)_([^.]+)\.md", filename)
    if match:
        topic_id = match.group(1)
        slug = match.group(2)
        return slug, topic_id
    return None, None

# --- Chunk Markdown Notes ---
def split_course_markdown(text):
    sections = re.split(r"^#+\s+", text, flags=re.MULTILINE)
    chunks = []
    for section in sections:
        paras = section.strip().split("\n\n")
        current, current_len = [], 0
        for para in paras:
            tokens = len(enc.encode(para))
            if current_len + tokens > CHUNK_SIZE:
                chunk = "\n\n".join(current)
                chunks.append(chunk)
                current = [para]
                current_len = tokens
            else:
                current.append(para)
                current_len += tokens
        if current:
            chunks.append("\n\n".join(current))
    return chunks

# --- Chunk Discourse Posts ---
def chunk_discourse_file(text):
    chunks = []
    post_blocks = re.split(r"-{3,}\n\*{2}", text)
    for post in post_blocks:
        post = post.strip()
        if not post:
            continue
        header_match = re.match(r"(.*?)\*\* posted on (.*?):\n\n", post)
        if header_match:
            meta = header_match.group(0)
            body = post[len(meta):].strip()
            chunks.append(f"{meta}{body}")
        else:
            chunks.append(post)
    return chunks

# --- Split Large Chunks ---
def split_large_chunk(text, max_tokens=SUBCHUNK_TOKENS):
    words = text.split()
    subchunks = []
    for i in range(0, len(words), max_tokens - OVERLAP):
        sub = " ".join(words[i:i + max_tokens])
        if sub:
            subchunks.append(sub)
    return subchunks

# --- Get Embedding with Retry ---
def get_embedding(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            tokens = len(enc.encode(text))
            if tokens > 8192:
                print(f"⚠️ Chunk too large ({tokens} tokens). Splitting...")
                embeddings = []
                for sub in split_large_chunk(text):
                    sub_emb = get_embedding(sub)
                    if sub_emb:
                        embeddings.append((sub, sub_emb))
                return embeddings if embeddings else None

            response = client.embeddings.create(
                model=MODEL,
                input=text,
                dimensions=512
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️ Failed to embed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
    return None

# --- Main Embedding Generator ---
if __name__ == "__main__":
    folders = ["tds_markdown", "tds_discourse_md"]
    all_chunks, all_embeddings, metas = [], [], []

    files = []
    for folder in folders:
        files.extend(Path(folder).rglob("*.md"))

    for file_path in tqdm(files, desc="Processing files"):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        is_discourse = "tds_discourse_md" in str(file_path)
        chunks = chunk_discourse_file(text) if is_discourse else split_course_markdown(text)

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding:
                if isinstance(embedding, list) and isinstance(embedding[0], tuple):
                    for j, (subchunk, emb) in enumerate(embedding):
                        meta = {
                            "filename": str(file_path),
                            "chunk_id": f"{i}_{j}",
                            "text": subchunk[:200].replace("\n", " ")
                        }
                        if is_discourse:
                            slug, topic_id = extract_slug_id(file_path.name)
                            if slug and topic_id:
                                meta["url"] = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}/{i}"
                        else:
                            slug = slugify(file_path.stem)
                            meta["url"] = f"https://tds.s-anand.net/#/{slug}"
                        all_chunks.append(subchunk)
                        all_embeddings.append(emb)
                        metas.append(meta)
                else:
                    meta = {
                        "filename": str(file_path),
                        "chunk_id": i,
                        "text": chunk[:200].replace("\n", " ")
                    }
                    if is_discourse:
                        slug, topic_id = extract_slug_id(file_path.name)
                        if slug and topic_id:
                            meta["url"] = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}/{i}"
                    else:
                        slug = slugify(file_path.stem)
                        meta["url"] = f"https://tds.s-anand.net/#/{slug}"
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    metas.append(meta)

    np.savez_compressed(OUTPUT_FILE, chunks=all_chunks, embeddings=np.array(all_embeddings), metadata=metas)
    print(f"✅ Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")
