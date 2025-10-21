#!/usr/bin/env python3
"""
vaio/core/description.py
========================
Stage 3 – TD (Title + Description) Generation - DE-HARDCODED VERSION
"""

from __future__ import annotations

import re
import sys
from typing import Dict, Tuple, List
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
    TMP_FILENAME,
    # New constants
    TD_MAX_CAPTION_LENGTH,
    TD_MAX_TITLE_LENGTH,
    TD_TITLE_TRUNCATE_LENGTH,
    TD_MAX_HASHTAGS,
    TD_HASHTAG_MIX,
    TEMPLATE_COMMENT_PREFIX,
    TEMPLATE_BLOCK_PATTERN,
    BLOCK_NAMES,
    GUIDANCE_BLOCKS,
    DEFAULT_HASHTAGS,
    DEFAULT_DESCRIPTION_GUIDE
)
from .utils import read_text, write_text, load_meta, save_meta, confirm, ensure_dir
from vaio.kb.query import inject_context, build_if_needed, load_kb_if_available, _resolve_kb_dir_for_video
from vaio.kb.store import collection_stats

# Use constant for pattern
TAG_BLOCK_PATTERN = re.compile(TEMPLATE_BLOCK_PATTERN, re.DOTALL | re.IGNORECASE)

def parse_template_advanced(text: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Advanced template parsing that preserves EXACT structure.
    """
    # Remove ONLY comment lines using constant
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(TEMPLATE_COMMENT_PREFIX) and not stripped.startswith("---"):
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
            if content == BLOCK_NAMES['VIDEO_DESCRIPTION']:
                result_parts.append(generated_description)
            elif content == BLOCK_NAMES['HASH_TAGS']:
                result_parts.append(generated_hashtags)
            elif content in GUIDANCE_BLOCKS:
                # Remove guidance blocks from final output
                continue
            else:
                # Keep other blocks as-is
                result_parts.append(blocks.get(content, ""))
    
    return "\n\n".join(part for part in result_parts if part.strip())

# ────────────────────────────────
# 🧠 PROMPTS SECTION
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
    f"• {TD_MAX_TITLE_LENGTH-30}-{TD_MAX_TITLE_LENGTH-20} characters ideal\n"
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
    "## VIDEO CONTEXT:\n"
    "{captions}\n\n"
    "Create a NEW title that captures the same essence but is original and optimized."
)

# ────────────────────────────────
# 🏷️ HASHTAG GENERATION PROMPTS
# ────────────────────────────────

SYSTEM_PROMPT_HASHTAGS = (
    "You are a YouTube growth expert specializing in hashtag optimization. "
    "Create strategic hashtag mixes that balance reach and relevance. "
    "Output ONLY the hashtags as a single line, no explanations."
)

USER_PROMPT_HASHTAGS = (
    "Generate strategic YouTube hashtags inspired by these guidelines:\n\n"
    "## CONTEXT:\n"
    "Title: {title}\n"
    "Description: {description}\n\n"
    "## HASHTAG INSPIRATION (use as style reference):\n"
    "{hashtag_guidelines}\n\n"
    "## REQUIREMENTS:\n"
    "- Create FRESH hashtags in the same style/theme\n"
    "- Mix of broad ({broad_min}-{broad_max}), niche ({niche_min}-{niche_max}), specific ({specific_min}-{specific_max})\n"
    "- Include primary keywords from content\n"
    "- Proper CamelCase formatting\n"
    "- Maximum {max_hashtags} tags total\n"
    "- Ensure all tags are relevant\n\n"
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
    template_structure: List[Tuple[str, str]]

# ────────────────────────────────
# 📥 LOAD INPUTS
# ────────────────────────────────
def load_inputs(video_path: Path, template_path: Path | None = None) -> InputData:
    """Loads inputs with advanced template parsing."""
    
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
# 🧠 CONTENT GENERATION
# ────────────────────────────────

def generate_description(video_path: Path, data: InputData) -> str:
    """Generate completely fresh description content."""
    from vaio.kb.query import inject_context

    print("🧠 Generating optimized description content...")

    # Extract guidance using constants
    instructions = data.blocks.get(BLOCK_NAMES['INSTRUCTIONS'], "")
    context = data.blocks.get(BLOCK_NAMES['CONTEXT'], "")
    desc_guidelines = data.blocks.get(BLOCK_NAMES['VIDEO_DESCRIPTION'], DEFAULT_DESCRIPTION_GUIDE)
    name_hint = data.blocks.get(BLOCK_NAMES['VIDEO_NAME'], "")

    # Build prompt for ORIGINAL content
    desc_prompt = USER_PROMPT_DESC.format(
        desc_guidelines=desc_guidelines,
        context=context,
        captions=data.captions[:TD_MAX_CAPTION_LENGTH] if data.captions else "(No captions available)",
        instructions=instructions
    )

    # Add name hint for context
    if name_hint:
        desc_prompt = f"## PRODUCT/VIDEO FOCUS:\n{name_hint}\n\n{desc_prompt}"

    # Enhance with KB context
    desc_prompt = inject_context(video_path, desc_prompt, top_k=3, task="desc")

    # Generate FRESH content
    description = chat_with_retries(
        OLLAMA_MODEL, 
        SYSTEM_PROMPT_DESC.format(src_lang=SOURCE_LANGUAGE), 
        desc_prompt
    )

    # Clean any accidental markers
    description = re.sub(r'<!--.*?-->', '', description, flags=re.DOTALL)
    description = re.sub(r'##\s+\w+', '', description)
    
    return description.strip()

def generate_title(video_path: Path, data: InputData) -> str:
    """Generate optimized title using all context."""
    from vaio.kb.query import inject_context

    print("🧠 Generating optimized title...")

    # Extract guidance using constants
    instructions = data.blocks.get(BLOCK_NAMES['INSTRUCTIONS'], "")
    context = data.blocks.get(BLOCK_NAMES['CONTEXT'], "")
    name_hint = data.blocks.get(BLOCK_NAMES['VIDEO_NAME'], "")
    desc_guidelines = data.blocks.get(BLOCK_NAMES['VIDEO_DESCRIPTION'], "")

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
    
    # Clean and validate using constants
    title = re.sub(r'[#"]', '', title).strip()
    if len(title) > TD_MAX_TITLE_LENGTH:
        title = title[:TD_TITLE_TRUNCATE_LENGTH] + "..."
    
    return title

def optimize_hashtags(video_path: Path, hashtag_guidelines: str, description: str, title: str) -> str:
    """Generate optimized hashtags that preserve guideline essence."""
    from vaio.kb.query import inject_context
    
    # Build hashtag strategy description from constants
    broad_min, broad_max = TD_HASHTAG_MIX['broad']
    niche_min, niche_max = TD_HASHTAG_MIX['niche']
    specific_min, specific_max = TD_HASHTAG_MIX['specific']
    
    # Use the new prompt templates from the Prompts section
    hashtag_prompt = USER_PROMPT_HASHTAGS.format(
        title=title,
        description=description[:1000],
        hashtag_guidelines=hashtag_guidelines,
        broad_min=broad_min,
        broad_max=broad_max,
        niche_min=niche_min,
        niche_max=niche_max,
        specific_min=specific_min,
        specific_max=specific_max,
        max_hashtags=TD_MAX_HASHTAGS
    )

    try:
        hashtag_prompt = inject_context(video_path, hashtag_prompt, top_k=2, task="desc")
        
        response = chat_with_retries(
            OLLAMA_MODEL,
            SYSTEM_PROMPT_HASHTAGS,
            hashtag_prompt
        )
        
        # Clean and validate
        hashtags = ' '.join([
            tag.strip() for tag in response.split() 
            if tag.strip().startswith('#') and len(tag.strip()) > 2
        ])
        
        return hashtags or DEFAULT_HASHTAGS
        
    except Exception as e:
        print(f"⚠️ Hashtag optimization failed: {e}")
        # Fallback to cleaned original guidelines
        return ' '.join([
            tag.strip() for tag in hashtag_guidelines.split() 
            if tag.strip().startswith('#')
        ]) or DEFAULT_HASHTAGS

# ────────────────────────────────
# 🧠 KNOWLEDGE BASE
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
# 🚀 MAIN ORCHESTRATOR
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
    fresh_description = generate_description(video_path, data)
    optimized_title = generate_title(video_path, data)
    
    # Generate optimized hashtags
    print("🔖 Generating strategic hashtags...")
    hashtag_guidelines = data.blocks.get(BLOCK_NAMES['HASH_TAGS'], "")
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