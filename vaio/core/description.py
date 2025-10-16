#!/usr/bin/env python3
"""
vaio/core/description.py
========================
Stage 3 – TD (Title + Description) Generation

Generates an SEO-optimized title and description pair (TD) from verified captions
and an optional template file. The resulting file is saved as:

  description/td.<lang_code>.txt

Usage:
  vaio desc <video.mp4> [--template-file td_temp.txt]
"""

from __future__ import annotations
from pathlib import Path
import sys
import ollama

from .constants import (
    SOURCE_LANGUAGE,
    OLLAMA_MODEL,
    TEMPERATURE,
    MAX_RETRIES,
    INITIAL_BACKOFF_S,
    META_FILENAME,
)
from .utils import read_text, write_text, load_meta, save_meta, confirm, ensure_dir


# ────────────────────────────────
# 🧠 PROMPTS
# ────────────────────────────────
SYSTEM_PROMPT_TITLE = (
    "You are an expert in YouTube SEO, media marketing, and title optimization. "
    "Your task is to analyze captions (SRT) and produce a concise, catchy, and SEO-optimized title. "
    "Rules:\n"
    "1) Reflect the video's main theme.\n"
    "2) Be emotional, attractive, and relevant to real search behavior.\n"
    "3) Output ONLY the title text, no comments."
)

USER_PROMPT_TITLE = (
    "Analyze the following subtitles (SRT) and generate a compelling YouTube title in {src_lang}.\n\n"
    "----- SRT CONTENT -----\n{captions}\n----- END SRT -----"
)

SYSTEM_PROMPT_DESC = (
    "You are an expert in multilingual SEO and YouTube optimization. "
    "Generate a full video description in {src_lang} using the captions and given template.\n"
    "Rules:\n"
    "1) Replace <Hook and SEO optimized video description from captions> with your generated text.\n"
    "2) Preserve formatting, structure, and other static text.\n"
    "3) Output ONLY the final description text."
)

USER_PROMPT_DESC = (
    "Use the following captions and layout template to write the full description.\n\n"
    "----- CAPTIONS -----\n{captions}\n----- END CAPTIONS -----\n\n"
    "----- TEMPLATE -----\n{template}\n----- END TEMPLATE -----"
)


# ────────────────────────────────
# 🔁 CORE CHAT LOGIC
# ────────────────────────────────
def chat_once(model: str, system_prompt: str, user_prompt: str) -> str:
    """Perform a single Ollama chat completion."""
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": TEMPERATURE},
        stream=False,
    )
    return (resp.get("message") or {}).get("content", "").strip()


def chat_with_retries(model: str, system_prompt: str, user_prompt: str) -> str:
    delay = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return chat_once(model, system_prompt, user_prompt)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"⚠️ Retry {attempt}/{MAX_RETRIES} after error: {e}")
            import time; time.sleep(delay)
            delay *= 2
    raise RuntimeError("Ollama failed after all retries.")


# ────────────────────────────────
# 🧩 TD GENERATION
# ────────────────────────────────
def process(video_path: Path, template_path: Path | None = None):
    """
    Generates title+description file based on SRT captions and optional template.
    Produces: description/td.<lang>.txt
    """
    # Detect language dynamically
    lang_code = SOURCE_LANGUAGE.lower()
    captions_path = video_path.parent / "captions" / f"{video_path.stem}.{lang_code}.srt"
    if not captions_path.exists():
        print(f"⚠️ No '{lang_code}' caption found. Searching for any caption variant...")
        alt_srts = list((video_path.parent / "captions").glob(f"{video_path.stem}.*.srt"))
        if alt_srts:
            captions_path = alt_srts[0]
            lang_code = captions_path.suffixes[-2].lstrip(".")
            print(f"📄 Using fallback captions: {captions_path.name}")
        else:
            print(f"❌ No captions found for {video_path.name}. Run `vaio audio {video_path.name}` first.")
            sys.exit(1)


    captions = read_text(captions_path)

    if template_path and template_path.exists():
        template = read_text(template_path)
        print(f"📋 Using provided template: {template_path.name}")
    else:
        print("⚠️ No template file detected.")
        if not confirm("Continue without template and use default layout?"):
            print("❌ Aborted. Provide a template (e.g. td_temp.txt) and rerun:")
            print(f"   vaio desc \"{video_path.name}\" --template-file td_temp.txt")
            sys.exit(0)
        template = "<Hook and SEO optimized video description from captions>\n\n<Video description>"

    print("🧠 Generating SEO title...")
    title_prompt = USER_PROMPT_TITLE.format(src_lang=SOURCE_LANGUAGE, captions=captions)
    title = chat_with_retries(OLLAMA_MODEL, SYSTEM_PROMPT_TITLE, title_prompt)

    print("🧠 Generating SEO description...")
    desc_prompt = USER_PROMPT_DESC.format(captions=captions, template=template)
    desc = chat_with_retries(OLLAMA_MODEL, SYSTEM_PROMPT_DESC.format(src_lang=SOURCE_LANGUAGE), desc_prompt)

    # Combine output
    desc_dir = video_path.parent / "description"
    ensure_dir(desc_dir)
    td_path = desc_dir / f"td.{lang_code}.txt"
    combined = f"{title.strip()}\n\n\n{desc.strip()}\n"
    write_text(td_path, combined)

    print(f"✅ TD generated → {td_path}")
    print("🧩 Opening in VS Code for review...")

    import shutil, subprocess
    try:
        if shutil.which("code"):
            subprocess.Popen(["code", str(td_path)])
        else:
            print("⚠️ VS Code not found in PATH. Skipping auto-open.")
    except Exception as e:
        print(f"⚠️ Could not auto-open TD file: {e}")

    print()
    if not confirm("Is the title & description correct?"):
        print("❌ Aborted. Edit and rerun `vaio continue` after review.")
        sys.exit(0)

    meta = load_meta(video_path)
    meta["stage"] = "description_done"
    meta["td_lang"] = lang_code
    meta["td_file"] = str(td_path.name)

    save_meta(video_path, meta)

    print("✅ TD confirmed by user.")
    return td_path
