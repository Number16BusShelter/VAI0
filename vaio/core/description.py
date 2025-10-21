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

import re
import sys
from typing import Dict, Tuple
from pathlib import Path
import ollama

from .constants import (
    SOURCE_LANGUAGE,
    SOURCE_LANGUAGE_CODE,
    OLLAMA_MODEL,
    TEMPERATURE,
    MAX_RETRIES,
    INITIAL_BACKOFF_S,
    META_FILENAME,
    TMP_FILENAME
)
from .utils import read_text, write_text, load_meta, save_meta, confirm, ensure_dir
from vaio.kb.query import inject_context, build_if_needed, load_kb_if_available, _resolve_kb_dir_for_video
from vaio.kb.store import collection_stats

TAG_BLOCK_PATTERN = re.compile(
    r"<!--\s*<(?P<name>[^>]+)>\s*-->(?P<content>.*?)<!--\s*</\1>\s*-->",
    re.DOTALL | re.IGNORECASE
)

def parse_template_file(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Parses a VAIO template file.

    Returns:
        (clean_text, blocks)
        - clean_text: all non-comment, non-semantic text that should appear verbatim
        - blocks: dict of named blocks { "Video Name": "...", "Video Description": "...", "Hash tags": "..." }
    """
    # 1️⃣ Remove comment lines (start with "--")
    lines = [
        line for line in text.splitlines()
        if not line.strip().startswith("--")
    ]
    text = "\n".join(lines)

    # 2️⃣ Extract semantic blocks
    blocks = {}
    for match in TAG_BLOCK_PATTERN.finditer(text):
        name = match.group("name").strip()
        content = match.group("content").strip()
        blocks[name] = content

    # 3️⃣ Keep everything outside semantic blocks (verbatim content)
    clean_text = TAG_BLOCK_PATTERN.sub("", text).strip()

    return clean_text, blocks


# ────────────────────────────────
# 🧠 PROMPTS
# ────────────────────────────────
# ────────────────────────────────
# 🧠 PROMPTS
# ────────────────────────────────
SYSTEM_PROMPT_DESC = (
    "You are a professional YouTube content strategist, copywriter, and SEO expert. "
    "Your task is to generate a highly readable, SEO-optimized YouTube video description "
    "in {src_lang} using the provided captions, contextual notes, and structured template.\n\n"
    "Requirements:\n"
    "1) Use the captions to infer the main topic, tone, and key moments.\n"
    "2) Fill all relevant template placeholders with coherent, engaging text.\n"
    "3) Keep a professional tone and natural rhythm suitable for YouTube viewers.\n"
    "4) Do not invent unrelated topics; stay strictly within context.\n"
    "5) NEVER include metadata markers (like <!-- --> or ## sections) in the output.\n"
    "6) Preserve all non-dynamic parts of the template as-is.\n"
    "7) Output ONLY the final, ready-to-publish description text."
)

USER_PROMPT_DESC = (
    "Use the following materials to write a complete YouTube description:\n\n"
    "## Captions\n{captions}\n\n"
    "## Template\n{template}\n\n"
    "If relevant, include hashtags from context or template naturally at the end."
)

SYSTEM_PROMPT_TITLE = (
    "You are an expert in YouTube SEO and media marketing. "
    "Your task is to write a concise, compelling, and search-optimized title "
    "for a video, using the provided description, captions, and context.\n\n"
    "Rules:\n"
    "1) The title must reflect the video's actual topic and emotional core.\n"
    "2) Keep it under 100 characters if possible.\n"
    "3) Use capitalization consistent with YouTube standards (Title Case).\n"
    "4) No hashtags, emojis, or quotation marks unless critical for emphasis.\n"
    "5) Output ONLY the title text — no commentary or explanations."
)

USER_PROMPT_TITLE = (
    "Generate the most relevant, high-CTR YouTube title in {src_lang}, "
    "based on the video’s content below.\n\n"
    "## Captions\n{captions}\n\n"
    "## Description\n{description}\n\n"
    "Use natural language and strong search intent keywords."
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
# ────────────────────────────────
# 🧩 TD GENERATION (refactored)
# ────────────────────────────────

from dataclasses import dataclass


@dataclass
class InputData:
    lang_code: str
    captions: str
    raw_template: str
    blocks: dict[str, str]
    template_text: str


# ────────────────────────────────
# 📥 LOAD INPUTS
# ────────────────────────────────
def load_inputs(video_path: Path, template_path: Path | None = None) -> InputData:
    """Loads captions, template, and parses structured blocks."""

    lang_code = SOURCE_LANGUAGE_CODE.lower()
    captions = ""

    # Load captions
    captions_dir = video_path.parent / "captions"
    if captions_dir.exists():
        captions_path = captions_dir / f"{video_path.stem}.{lang_code}.srt"
        if not captions_path.exists():
            alt_srts = list(captions_dir.glob(f"{video_path.stem}.*.srt"))
            if alt_srts:
                captions_path = alt_srts[0]
                lang_code = captions_path.suffixes[-2].lstrip(".")
                print(f"📄 Using fallback captions: {captions_path.name}")
            else:
                print("⚠️ No captions found. Continuing without captions...")
                captions_path = None
        if captions_path and captions_path.exists():
            captions = read_text(captions_path)
    else:
        print("⚠️ No captions directory found. Continuing without captions...")

    # Load template
    if template_path and template_path.exists():
        raw_template = read_text(template_path)
        print(f"📋 Using provided template: {template_path.name}")
    else:
        candidates = [
            Path.resolve(video_path.parent / TMP_FILENAME),
            video_path.parent / TMP_FILENAME,
            Path.cwd() / TMP_FILENAME,
            Path.cwd() / "templates" / TMP_FILENAME,
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            print(f"📄 Found default template: {found}")
            raw_template = read_text(found)
        else:
            print("⚠️ No template file detected. Proceeding in captions-only mode.")
            raw_template = "<!-- <Video Description> -->\n<Hook>\n<!-- </Video Description> -->"

    # Parse template
    base_text, blocks = parse_template_file(raw_template)
    template_text = (
        (blocks.get("Video Description") or "") + "\n\n" + base_text
    ).strip() or "<Hook and SEO optimized video description from captions>"

    if blocks:
        print(f"🧱 Parsed template sections: {', '.join(blocks.keys())}")
    else:
        print("ℹ️ No structured template sections found (plain text mode).")

    return InputData(
        lang_code=lang_code,
        captions=captions,
        raw_template=raw_template,
        blocks=blocks,
        template_text=template_text,
    )


# ────────────────────────────────
# 🧠 KNOWLEDGE BASE
# ────────────────────────────────
def prepare_kb(video_path: Path):
    from vaio.kb.query import build_if_needed, load_kb_if_available, _resolve_kb_dir_for_video
    from vaio.kb.store import collection_stats

    try:
        build_if_needed(video_path)  # uses DEFAULT_KB_DIR internally when enabled
        kb_index = load_kb_if_available(video_path)
        kb_dir = _resolve_kb_dir_for_video(video_path)
        if kb_index and kb_dir and kb_dir.exists():
            stats = collection_stats(kb_dir)
            print(f"🧠 KB active: {stats['collection']} ({stats['count']} docs) dir={stats['dir']}")
            return stats
        print("⚠️ KB not loaded or empty. Continuing without contextual enrichment.")
    except Exception as e:
        print(f"⚠️ KB preparation skipped: {e}")



# ────────────────────────────────
# 🧠 DESCRIPTION GENERATION
# ────────────────────────────────
def generate_description(video_path: Path, data: InputData) -> str:
    """Generate SEO-optimized description text."""
    from vaio.kb.query import inject_context

    print("🧠 Generating SEO description...")

    desc_prompt = USER_PROMPT_DESC.format(captions=data.captions, template=data.template_text)
    desc_prompt = inject_context(video_path, desc_prompt, task="desc")

    inst = data.blocks.get("Instructions", "")
    ctx = data.blocks.get("Context", "")
    desc_section = data.blocks.get("Video Description", "")
    tags = data.blocks.get("Hash tags", "")

    # Structured LLM prompt
    desc_prompt = (
        f"## Instructions\n{inst}\n\n"
        f"## Context\n{ctx}\n\n"
        f"## Description Base\n{desc_section}\n\n"
        f"## Hashtags\n{tags}\n\n"
        f"{USER_PROMPT_DESC.format(captions=data.captions, template=data.template_text)}"
    )

    desc_prompt = inject_context(video_path, desc_prompt, task="desc")


    return chat_with_retries(OLLAMA_MODEL, SYSTEM_PROMPT_DESC.format(src_lang=SOURCE_LANGUAGE), desc_prompt)


# ────────────────────────────────
# 🧠 TITLE GENERATION
# ────────────────────────────────
def generate_title(video_path: Path, data: InputData, description: str | None = None) -> str:
    """Generate a title using description, captions, or both."""
    from vaio.kb.query import inject_context

    print("🧠 Generating SEO title...")

    # Choose best available context
    captions_part = data.captions.strip()
    desc_part = description.strip() if description else ""
    combined_context = ""

    if captions_part and desc_part:
        combined_context = f"## DESCRIPTION\n{desc_part}\n\n## CAPTIONS\n{captions_part}"
    elif desc_part:
        combined_context = f"## DESCRIPTION\n{desc_part}"
    elif captions_part:
        combined_context = f"## CAPTIONS\n{captions_part}"

    title_prompt = USER_PROMPT_TITLE.format(
        src_lang=SOURCE_LANGUAGE,
        captions=captions_part or "(no captions)",
        description=desc_part or "(no description)"
    ) + f"\n\n{combined_context}"

    title_prompt = inject_context(video_path, title_prompt, task="title")

    inst = data.blocks.get("Instructions", "")
    ctx = data.blocks.get("Context", "")
    name_hint = data.blocks.get("Video Name", "")

    if inst or ctx or name_hint:
        title_prompt = (
            f"## Instructions\n{inst}\n\n"
            f"## Context\n{ctx}\n\n"
            f"## Video Name Hint\n{name_hint}\n\n"
            f"{title_prompt}"
        )

    return chat_with_retries(OLLAMA_MODEL, SYSTEM_PROMPT_TITLE, title_prompt)


# ────────────────────────────────
# 💾 SAVE RESULTS
# ────────────────────────────────
def save_td(video_path: Path, title: str, description: str, lang_code: str):
    desc_dir = video_path.parent / "description"
    ensure_dir(desc_dir)
    td_path = desc_dir / f"td.{lang_code}.txt"

    combined = f"{title.strip()}\n\n\n{description.strip()}\n"
    write_text(td_path, combined)
    print(f"✅ TD generated → {td_path}")

    import shutil, subprocess
    try:
        if shutil.which("code"):
            subprocess.Popen(["code", str(td_path)])
        else:
            print("⚠️ VS Code not found in PATH. Skipping auto-open.")
    except Exception as e:
        print(f"⚠️ Could not auto-open TD file: {e}")

    meta = load_meta(video_path)
    meta.update({
        "stage": "description_done",
        "td_lang": lang_code,
        "td_file": str(td_path.name)
    })
    save_meta(video_path, meta)

    print("✅ TD metadata saved.")
    return td_path


# ────────────────────────────────
# 🚀 MAIN ORCHESTRATOR
# ────────────────────────────────
def process(video_path: Path, template_path: Path | None = None):
    """Main orchestrator for TD generation."""
    data = load_inputs(video_path, template_path)
    prepare_kb(video_path)

    desc = generate_description(video_path, data)
    title = generate_title(video_path, data, description=desc)

    td_path = save_td(video_path, title, desc, data.lang_code)

    print()
    if not confirm("Is the title & description correct?"):
        print("❌ Aborted. Edit and rerun `vaio continue` after review.")
        sys.exit(0)

    print("✅ TD confirmed by user.")
    return td_path

