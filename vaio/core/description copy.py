#!/usr/bin/env python3
"""
vaio/core/description.py
========================
Stage 3 – TD (Title + Description) Generation
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

def parse_template_advanced(text: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Advanced template parsing that preserves EXACT structure.
    
    Returns:
        - blocks: dict of semantic block content
        - template_structure: list of (content_type, content) preserving order
          where content_type is either "block" or "verbatim"
    """
    # Remove ONLY comment lines (starting with --), keep everything else
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") and not stripped.startswith("---"):
            continue  # Skip comment lines
        lines.append(line)
    cleaned_text = "\n".join(lines)
    
    # Extract blocks while preserving positions
    blocks = {}
    template_structure = []
    last_pos = 0
    
    for match in TAG_BLOCK_PATTERN.finditer(cleaned_text):
        # Add verbatim content before this block
        verbatim_before = cleaned_text[last_pos:match.start()].strip()
        if verbatim_before:
            template_structure.append(("verbatim", verbatim_before))
        
        # Add the block
        name = match.group("name").strip()
        content = match.group("content").strip()
        blocks[name] = content
        template_structure.append(("block", name))
        
        last_pos = match.end()
    
    # Add remaining verbatim content
    verbatim_after = cleaned_text[last_pos:].strip()
    if verbatim_after:
        template_structure.append(("verbatim", verbatim_after))
    
    return blocks, template_structure

def reconstruct_template(blocks: Dict[str, str], template_structure: List[Tuple[str, str]], 
                        generated_description: str, generated_hashtags: str) -> str:
    """
    Reconstruct the template with generated content while preserving EXACT structure.
    """
    result_parts = []
    
    for content_type, content in template_structure:
        if content_type == "verbatim":
            # Preserve verbatim content EXACTLY
            result_parts.append(content)
        elif content_type == "block":
            if content == "Video Description":
                result_parts.append(generated_description)
            elif content == "Hash tags":
                result_parts.append(generated_hashtags)
            elif content == "Video Name":
                # Video Name block is handled separately for title generation
                continue  # Remove from final output
            elif content in ["Instructions", "Context"]:
                # Remove guidance blocks from final output
                continue
            else:
                # Keep other blocks as-is
                result_parts.append(blocks.get(content, ""))
    
    return "\n\n".join(part for part in result_parts if part.strip())

# ────────────────────────────────
# 🧠 PROMPTS
# ────────────────────────────────

SYSTEM_PROMPT_DESC = (
    "You are a professional YouTube content strategist. "
    "Generate FRESH, ORIGINAL description content that follows the GUIDELINES but is NOT copied.\n\n"
    "CRITICAL RULES:\n"
    "• Use the 'Video Description' block as STYLE/TONE GUIDELINES only\n" 
    "• Create COMPLETELY NEW content in the same style\n"
    "• DO NOT copy sentences or phrases from the guidelines\n"
    "• Expand on the guidelines with new, engaging content\n"
    "• Keep the same emotional tone and professional level\n"
    "• NEVER include block markers or comments\n"
    "• Output ONLY the new description content\n"
)

USER_PROMPT_DESC = (
    "Create ORIGINAL YouTube description content in this style:\n\n"
    "## STYLE GUIDELINES (DO NOT COPY):\n"
    "{desc_guidelines}\n\n"
    "## BRAND CONTEXT:\n"
    "{context}\n\n"
    "## VIDEO CONTEXT:\n"
    "{captions}\n\n"
    "## SPECIAL INSTRUCTIONS:\n"
    "{instructions}\n\n"
    "Generate FRESH content that captures the same essence but uses different wording and expands on the ideas."
)

SYSTEM_PROMPT_TITLE = (
    "You are a YouTube SEO expert. Create compelling, search-optimized titles.\n\n"
    "RULES:\n"
    "• Use 'Video Name' as INSPIRATION, not a template\n"
    "• Create NEW titles that capture the same essence\n"
    "• 60-70 characters ideal\n"
    "• Title Case, no hashtags/emojis\n"
    "• Reflect core topic and emotional appeal\n"
    "• Output ONLY the title text"
)

USER_PROMPT_TITLE = (
    "Generate a high-CTR YouTube title inspired by:\n\n"
    "## NAME INSPIRATION:\n"
    "{name_hint}\n\n"
    "## DESCRIPTION STYLE:\n"
    "{desc_guidelines}\n\n"
    "## BRAND CONTEXT:\n"
    "{context}\n\n"
    "## VIDEO CONTENT:\n"
    "{captions}\n\n"
    "Create a NEW title that captures the same essence but is original and optimized."
)

TAG_GENERATION_PROMPT= (
    "Generate strategic YouTube hashtags inspired by these guidelines:"
    "## CONTEXT:"
    "Title: {title}"
    "Description: {description[:1000]}"
    "## HASHTAG INSPIRATION (use as style reference):"
    "{hashtag_guidelines}"
    "## REQUIREMENTS:"
    "- Create FRESH hashtags in the same style/theme"
    "- Mix of broad (3-4), niche (5-7), specific (2-3)"
    "- Include primary keywords from content"
    "- Proper CamelCase formatting"
    "- Maximum 15 tags total"
    "- Ensure all tags are relevant"
    "Output ONLY the hashtags as a single line."
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
from dataclasses import dataclass

@dataclass
class InputData:
    lang_code: str
    captions: str
    raw_template: str
    blocks: dict[str, str]
    verbatim_content: str  # without semantic blocks and without comments
    template_without_comments: str  # the entire template without comments (but with semantic blocks)

# ────────────────────────────────
# 📥 LOAD INPUTS
# ────────────────────────────────
def load_inputs(video_path: Path, template_path: Path | None = None) -> InputData:
    """Loads inputs with advanced template parsing."""
    
    lang_code = SOURCE_LANGUAGE_CODE.lower()
    captions = ""

    # Load captions (existing logic)
    captions_dir = video_path.parent / "captions"
    if captions_dir.exists():
        captions_path = captions_dir / f"{video_path.stem}.{lang_code}.srt"
        if not captions_path.exists():
            alt_srts = list(captions_dir.glob(f"{video_path.stem}.*.srt"))
            if alt_srts:
                captions_path = alt_srts[0]
                detected_lang = captions_path.suffixes[-2].lstrip(".")
                print(f"🌐 Using {detected_lang} captions: {captions_path.name}")
                lang_code = detected_lang
        if captions_path and captions_path.exists():
            captions = read_text(captions_path)
            print(f"📄 Loaded captions: {len(captions)} characters")

    # Load template
    template_candidates = []
    if template_path and template_path.exists():
        template_candidates = [template_path]
    else:
        template_candidates = [
            video_path.parent / TMP_FILENAME,
            Path.cwd() / TMP_FILENAME,
            Path.cwd() / "templates" / TMP_FILENAME,
        ]
    
    raw_template = ""
    for candidate in template_candidates:
        if candidate.exists():
            raw_template = read_text(candidate)
            print(f"📋 Using template: {candidate}")
            break
    
    if not raw_template:
        print("⚠️ No template file found. Using minimal defaults.")
        raw_template = ""

    # Advanced template parsing
    blocks, template_structure = parse_template_advanced(raw_template)
    
    if blocks:
        print(f"🧱 Semantic blocks found: {', '.join(blocks.keys())}")
    else:
        print("ℹ️ No semantic blocks found")

    return InputData(
        lang_code=lang_code,
        captions=captions,
        raw_template=raw_template,
        blocks=blocks,
        template_structure=template_structure,
    )


# ────────────────────────────────
# 🧠 ENHANCED CONTENT GENERATION
# ────────────────────────────────

def generate_fresh_description(video_path: Path, data: InputData) -> str:
    """Generate completely fresh description content."""
    from vaio.kb.query import inject_context

    print("🧠 Generating FRESH description content...")

    # Extract guidance
    instructions = data.blocks.get("Instructions", "")
    context = data.blocks.get("Context", "")
    desc_guidelines = data.blocks.get("Video Description", "")
    name_hint = data.blocks.get("Video Name", "")

    # Build prompt for ORIGINAL content
    desc_prompt = USER_PROMPT_DESC.format(
        desc_guidelines=desc_guidelines,
        context=context,
        captions=data.captions[:1500] if data.captions else "(No captions available)",
        instructions=instructions
    )

    # Add name hint for context
    if name_hint:
        desc_prompt = f"## PRODUCT/VIDEO FOCUS:\n{name_hint}\n\n{desc_prompt}"

    # Enhance with KB context
    desc_prompt = inject_context(video_path, desc_prompt, top_k=3, task="desc")

    # Generate FRESH content
    fresh_description = chat_with_retries(
        OLLAMA_MODEL, 
        SYSTEM_PROMPT_DESC.format(src_lang=SOURCE_LANGUAGE), 
        desc_prompt
    )

    # Clean any accidental markers
    fresh_description = re.sub(r'<!--.*?-->', '', fresh_description, flags=re.DOTALL)
    fresh_description = re.sub(r'##\s+\w+', '', fresh_description)
    
    return fresh_description.strip()


def generate_optimized_title(video_path: Path, data: InputData) -> str:
    """Generate optimized title using all context."""
    from vaio.kb.query import inject_context

    print("🧠 Generating optimized title...")

    # Extract guidance
    instructions = data.blocks.get("Instructions", "")
    context = data.blocks.get("Context", "")
    name_hint = data.blocks.get("Video Name", "")
    desc_guidelines = data.blocks.get("Video Description", "")

    # Prepare content context
    captions_preview = data.captions[:800] if data.captions else "(No captions)"

    title_prompt = USER_PROMPT_TITLE.format(
        name_hint=name_hint,
        desc_guidelines=desc_guidelines,
        context=context,
        captions=captions_preview,
        instructions=instructions
    )

    # Enhance with KB context
    title_prompt = inject_context(video_path, title_prompt, top_k=2, task="title")

    title = chat_with_retries(OLLAMA_MODEL, SYSTEM_PROMPT_TITLE, title_prompt)
    
    # Clean and validate
    title = re.sub(r'[#"]', '', title).strip()
    if len(title) > 100:
        title = title[:97] + "..."
    
    return title


def optimize_hashtags(video_path: Path, hashtag_guidelines: str, description: str, title: str) -> str:
    """Generate optimized hashtags that preserve guideline essence."""
    from vaio.kb.query import inject_context
    
    hashtag_prompt = TAG_GENERATION_PROMPT

    try:
        hashtag_prompt = inject_context(video_path, hashtag_prompt, top_k=2, task="desc")
        
        response = chat_with_retries(
            OLLAMA_MODEL,
            "You are a YouTube growth expert. Create strategic hashtag mixes. Output ONLY hashtags.",
            hashtag_prompt
        )
        
        # Clean and validate
        hashtags = ' '.join([
            tag.strip() for tag in response.split() 
            if tag.strip().startswith('#') and len(tag.strip()) > 2
        ])
        
        return hashtags or "#YouTube #Content"
        
    except Exception as e:
        print(f"⚠️ Hashtag optimization failed: {e}")
        # Fallback to cleaned original guidelines
        return ' '.join([
            tag.strip() for tag in hashtag_guidelines.split() 
            if tag.strip().startswith('#')
        ]) or "#YouTube #Content"
    
# Update the generate_description function to use optimized hashtags:
def generate_description(video_path: Path, data: InputData) -> str:
    """Generate SEO-optimized description text with proper structure."""
    from vaio.kb.query import inject_context

    print("🧠 Generating SEO description...")

    # Extract all semantic blocks
    instructions = data.blocks.get("Instructions", "")
    context = data.blocks.get("Context", "")
    desc_guidelines = data.blocks.get("Video Description", "")
    hashtag_guidelines = data.blocks.get("Hash tags", "")
    name_hint = data.blocks.get("Video Name", "")

    # Build comprehensive description prompt
    desc_prompt = USER_PROMPT_DESC.format(
        captions=data.captions[:2000] if data.captions else "(No captions available)",
        verbatim_content=data.verbatim_content,
        desc_guidelines=desc_guidelines,
        context=context,
        instructions=instructions
    )

    # Add name hint if available
    if name_hint and name_hint.strip():
        desc_prompt = f"## VIDEO NAME INSPIRATION\n{name_hint}\n\n{desc_prompt}"

    # Enhance with KB context
    desc_prompt = inject_context(video_path, desc_prompt, top_k=3, task="desc")

    # Generate description content
    description_content = chat_with_retries(
        OLLAMA_MODEL, 
        SYSTEM_PROMPT_DESC.format(src_lang=SOURCE_LANGUAGE), 
        desc_prompt
    )

    # Clean up any accidental metadata in the response
    description_content = re.sub(r'<!--.*?-->', '', description_content, flags=re.DOTALL)
    description_content = re.sub(r'##\s+\w+', '', description_content)
    description_content = description_content.strip()

    return description_content

# ────────────────────────────────
# 🧠 TITLE GENERATION
# ────────────────────────────────
def generate_title(video_path: Path, data: InputData, description: str | None = None) -> str:
    """Generate optimized title using all available context."""
    from vaio.kb.query import inject_context

    print("🧠 Generating SEO title...")

    # Extract all relevant guidance
    instructions = data.blocks.get("Instructions", "")
    context = data.blocks.get("Context", "")
    name_hint = data.blocks.get("Video Name", "")
    desc_guidelines = data.blocks.get("Video Description", "")

    # Prepare content context
    captions_preview = data.captions[:1000] if data.captions else "(No captions)"
    description_preview = description[:500] if description else desc_guidelines[:500]

    title_prompt = USER_PROMPT_TITLE.format(
        src_lang=SOURCE_LANGUAGE,
        captions=captions_preview,
        desc_guidelines=desc_guidelines,
        context=context,
        name_hint=name_hint,
        instructions=instructions
    )

    # Add description context if available
    if description:
        title_prompt = f"## GENERATED DESCRIPTION\n{description_preview}\n\n{title_prompt}"

    # Enhance with KB context
    title_prompt = inject_context(video_path, title_prompt, top_k=2, task="title")

    title = chat_with_retries(OLLAMA_MODEL, SYSTEM_PROMPT_TITLE, title_prompt)
    
    # Clean and validate title
    title = re.sub(r'[#"]', '', title).strip()
    
    # Ensure proper length
    if len(title) > 100:
        title = title[:97] + "..."
    
    return title

# ────────────────────────────────
# 💾 SAVE RESULTS
# ────────────────────────────────


# ────────────────────────────────
# 🚀 CORRECTED MAIN ORCHESTRATOR
# ────────────────────────────────

def process(video_path: Path, template_path: Path | None = None):
    """Main orchestrator with correct template handling."""
    print("🚀 Starting TD Generation with Advanced Template Processing...")
    
    # Load inputs with advanced parsing
    data = load_inputs(video_path, template_path)
    
    # Prepare knowledge base
    prepare_kb(video_path)
    
    print("\n📝 Generating optimized content...")
    
    # Generate FRESH content (not copied from guidelines)
    fresh_description = generate_fresh_description(video_path, data)
    optimized_title = generate_optimized_title(video_path, data)
    
    # Generate optimized hashtags
    print("🔖 Generating strategic hashtags...")
    hashtag_guidelines = data.blocks.get("Hash tags", "")
    optimized_hashtags = optimize_hashtags(
        video_path, hashtag_guidelines, fresh_description, optimized_title
    )
    
    # Reconstruct final description with EXACT template structure
    final_description = reconstruct_template(
        data.blocks,
        data.template_structure,
        fresh_description,
        optimized_hashtags
    )
    
    # Save results
    td_path = save_td(video_path, optimized_title, final_description, data.lang_code)

    # Preview
    print(f"\n📊 GENERATED CONTENT:")
    print(f"Title: {optimized_title} ({len(optimized_title)} chars)")
    print(f"Description: {len(final_description)} chars")
    print(f"Hashtags: {optimized_hashtags}")

    # User confirmation
    print()
    if not confirm("Is the title & description correct?"):
        print("❌ Aborted. Edit and rerun `vaio continue` after review.")
        sys.exit(0)

    print("✅ TD confirmed by user.")
    return td_path


# ────────────────────────────────
# 🧠 KNOWLEDGE BASE (keep existing)
# ────────────────────────────────
def prepare_kb(video_path: Path):
    from vaio.kb.query import build_if_needed, load_kb_if_available, _resolve_kb_dir_for_video
    from vaio.kb.store import collection_stats

    try:
        build_if_needed(video_path)
        kb_index = load_kb_if_available(video_path)
        kb_identifier = _resolve_kb_dir_for_video(video_path)
        
        if kb_index and kb_identifier:
            stats = collection_stats(kb_identifier)
            print(f"🧠 KB active: {stats['collection']} ({stats['count']} documents)")
            return stats
        print("⚠️ KB not loaded or empty. Continuing without contextual enrichment.")
    except Exception as e:
        print(f"⚠️ KB preparation skipped: {e}")


# ────────────────────────────────
# 🔁 CORE CHAT LOGIC (keep existing)
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
# 💾 SAVE RESULTS (keep existing)
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