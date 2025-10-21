from pathlib import Path
from vaio.kb import inject_context, build_if_needed

def run_llm_with_context(prompt: str, video_path: Path, **llm_kwargs):
    # Build index on-demand if empty (safe & quick when already built)
    build_if_needed(video_path)
    prompt = inject_context(video_path, prompt, top_k=3)
    # ... call your existing Ollama/LLM here ...
    # return ollama.generate(model="...", prompt=prompt, **llm_kwargs)
from pathlib import Path
from vaio.kb import inject_context, build_if_needed

def run_llm_with_context(prompt: str, video_path: Path, **llm_kwargs):
    # Build index on-demand if empty (safe & quick when already built)
    build_if_needed(video_path)
    prompt = inject_context(video_path, prompt, top_k=3)
    # ... call your existing Ollama/LLM here ...
    # return ollama.generate(model="...", prompt=prompt, **llm_kwargs)
