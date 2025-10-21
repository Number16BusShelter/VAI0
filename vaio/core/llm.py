# vaio/core/llm.py
from pathlib import Path
from vaio.kb import inject_context, build_if_needed
import ollama

def run_llm_with_context(prompt: str, video_path: Path, model: str = "qwen2.5:7b", **gen_kwargs) -> str:
    """
    Build KB if needed, inject top-K context, call Ollama, return .text
    """
    build_if_needed(video_path)
    enriched = inject_context(video_path, prompt, top_k=3)
    resp = ollama.generate(model=model, prompt=enriched, options=gen_kwargs or {})
    # Ollama returns dict with 'response' or 'message'
    return resp.get("response") or resp.get("message", {}).get("content", "")
