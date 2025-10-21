"""
vaio/core/translate.py
======================
Handles translation of title + description (TD) files into multiple target languages.
Runs after the base td.<src_lang>.txt file is verified.

Each output file is saved beside the source, named:
    td.<lang_code>.txt
"""

from __future__ import annotations
import concurrent.futures as cf
import sys
import time
from pathlib import Path

import ollama
from .constants import (
    SOURCE_LANGUAGE,
    SOURCE_LANGUAGE_CODE,
    TARGET_LANGUAGES,

    OLLAMA_MODEL,
    TEMPERATURE,
    MAX_RETRIES,
    INITIAL_BACKOFF_S,
    CONCURRENCY,
    META_FILENAME,
)
from .utils import read_text, write_text, load_meta, save_meta


# ────────────────────────────────
# 🧠 Base Translation Logic
# ────────────────────────────────
TRANSLATE_DESCRIPTION__PROMPT = (
    "You are an expert in multilingual SEO, localization, and cultural adaptation. "
    "Follow these strict rules:\n"
    "1) Output ONLY the final translated and SEO-optimized text.\n"
    "2) No explanations, comments, metadata, or notes.\n"
    "3) Preserve structure, formatting, and line breaks.\n"
    "4) Adapt naturally to cultural and linguistic norms.\n"
    "5) Use popular search phrases and idioms for natural SEO.\n"
    "6) Some languages distinguish between rough and cut diamonds. Always refer to 'cut diamonds', unless the rough is clearly described!\n"
)

USER_TRANSLATE_PROMPT_TEMPLATE = (
    "Translate and optimize the following TITLE and DESCRIPTION from {src_lang} to {tgt_lang}. "
    "Preserve layout and formatting exactly.\n\n"
    "----- BEGIN CONTENT -----\n{content}\n----- END CONTENT -----"
)


def chat_once(model: str, system_prompt: str, user_prompt: str) -> str:
    """Single Ollama chat call."""
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
            print(f"⚠️  Retry {attempt}/{MAX_RETRIES} after error: {e}")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Translation failed after all retries.")


# ────────────────────────────────
# 🌍 Translation Worker
# ────────────────────────────────
def translate_one(content: str, src_lang: str, code: str, lang: str, out_path: Path):
    """Translate one file and write result."""
    user_prompt = USER_TRANSLATE_PROMPT_TEMPLATE.format(
        src_lang=src_lang, tgt_lang=lang, content=content
    )
    try:
        translated = chat_with_retries(OLLAMA_MODEL, TRANSLATE_DESCRIPTION__PROMPT, user_prompt)
        if not translated.strip():
            raise ValueError("Empty model response")
        write_text(out_path, translated)
        print(f"✅ {lang} ({code}) → {out_path.name}")
        return True
    except Exception as e:
        print(f"❌ {lang} ({code}) error: {e}")
        return False


# ────────────────────────────────
# 🚀 Public Entry
# ────────────────────────────────
def process(video_path: Path):
    """
    Translate the approved td.<src_lang>.txt file into all TARGET_LANGUAGES.
    """
    # Load metadata to detect real TD file and language
    meta = load_meta(video_path)
    td_lang = meta.get("td_lang", SOURCE_LANGUAGE_CODE)

    # Primary location: description/td.<lang>.txt
    desc_dir = video_path.parent / "description"
    base_td = desc_dir / f"td.{td_lang}.txt"
    print(base_td)

    # Fallback search if not found
    if not base_td.exists():
        print(f"⚠️ No TD file found for language '{td_lang}' in {desc_dir}")
        candidates = list(desc_dir.glob("td.*.txt"))
        if candidates:
            base_td = candidates[0]
            td_lang = base_td.stem.split(".")[-1]
            print(f"📄 Using fallback TD: {base_td.name}")
        else:
            print("❌ No base TD file found anywhere.")
            print("Run `vaio desc` first to generate it.")
            sys.exit(1)


    print(f"🌍 Translating {base_td.name} into {len(TARGET_LANGUAGES)} languages…")
    content = read_text(base_td)
    results = {}

    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {}
        for code, lang in TARGET_LANGUAGES.items():
            out_file = desc_dir / f"td.{code}.txt"
            futures[ex.submit(translate_one, content, SOURCE_LANGUAGE, code, lang, out_file)] = (code, lang)

        for fut in cf.as_completed(futures):
            code, lang = futures[fut]
            results[code] = fut.result()

    # Update metadata
    meta["stage"] = "translated"
    meta["translations"] = {k: ("done" if v else "failed") for k, v in results.items()}
    meta["translated_from"] = td_lang
    save_meta(video_path, meta)


    print("\n✅ Translation stage complete.")
    print("Translated files saved beside the original video.")
    return results
