#!/usr/bin/env python3
"""
VAIO CLI
========
Video Auto Intelligence Operator — automated pipeline for:
 🎧 audio extraction
 💬 caption generation
 📝 SEO title & description creation
 🌐 translation & localization

Usage:
  vaio <command> [video.mp4] [options]
  vaio video.mp4         # auto-detect and run staged pipeline
"""

from __future__ import annotations
import argparse
import sys
import shutil
from pathlib import Path
from textwrap import dedent

from vaio.core import audio, description, translate, captions, full_auto
from vaio.core.utils import load_meta
from vaio.core.constants import PROJECT_VERSION, CLI_ICONS

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


# ────────────────────────────────
# Basic output helpers
# ────────────────────────────────
def info(msg): print(f"{CLI_ICONS.get('info','ℹ️')}  {msg}")
def success(msg): print(f"{CLI_ICONS.get('success','✅')} {msg}")
def warn(msg): print(f"{CLI_ICONS.get('warning','⚠️')} {msg}")
def fail(msg): print(f"{CLI_ICONS.get('error','❌')} {msg}")


# ────────────────────────────────
# Core command handlers
# ────────────────────────────────
def handle_audio(args): audio.process(Path(args.video_file).resolve())
def handle_desc(args):
    video = Path(args.video_file).resolve()
    template = Path(args.template_file).resolve() if args.template_file else None
    description.process(video, template)
def handle_translate(args): translate.process(Path(args.video_file).resolve())
def handle_captions(args): captions.process(Path(args.video_file).resolve())


def handle_continue(video_path: Path):
    meta = load_meta(video_path)
    stage = meta.get("stage", "init")
    info(f"Detected current stage: {stage}")

    # Add "audio_pending" and "init" to the same branch
    if stage in ("init", "audio_pending", "audio_done"):
        success("→ Running audio/caption verification...")
        audio.verify(video_path)
        stage = "captions_verified"

    if stage in ("captions_verified", "captions_done"):
        success("→ Generating title & description (TD)...")
        description.process(video_path)
        stage = "description_done"

    if stage == "description_done":
        success("→ Translating TD files...")
        translate.process(video_path)
        stage = "translated"

    if stage == "translated":
        success("→ Translating captions...")
        captions.process(video_path)
        stage = "captions_translated"

    success("🎉 All stages complete!")
    print("All outputs are stored beside the original video.")


import platform
import requests

def handle_debug(args: argparse.Namespace):
    """Run comprehensive environment & dependency diagnostics."""
    print("🧪 Running VAIO system diagnostics...\n")

    # --- Check functions ---
    def check_ffmpeg():
        return shutil.which("ffmpeg") is not None

    def check_whisper():
        try:
            import whisper  # noqa
            return True
        except Exception:
            return False

    def check_ollama():
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def check_vscode():
        return shutil.which("code") is not None

    def check_permissions():
        test_file = Path.cwd() / ".vaio_test.tmp"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            return True
        except Exception:
            return False

    # --- Collect results ---
    checks = {
        "Python Version": sys.version.split()[0],
        "OS": platform.system(),
        "FFmpeg": "✅ OK" if check_ffmpeg() else "❌ Missing",
        "Whisper": "✅ OK" if check_whisper() else "❌ Missing",
        "Ollama": "✅ Running" if check_ollama() else "❌ Not reachable",
        "VS Code": "✅ Found" if check_vscode() else "⚠️ Not found",
        "Write Access": "✅ OK" if check_permissions() else "❌ No permission",
    }

    # --- Display results ---
    if RICH_AVAILABLE:
        table = Table(title="VAIO Preflight Report", box=box.SIMPLE_HEAVY)
        table.add_column("Component", style="cyan", justify="right")
        table.add_column("Status", style="bold green")
        for key, val in checks.items():
            table.add_row(key, val)
        console.print(table)
    else:
        print("System Diagnostics:")
        for k, v in checks.items():
            print(f" - {k}: {v}")

    print("\n💡 Tip: Start Ollama with `ollama serve` if it's not running.")
    print("   Run `ollama list` to verify installed models.")
    print("   If any component is missing, reinstall via `pip install -r requirements.txt`.\n")

    # Exit code = 0 if all critical checks pass
    critical = all([
        check_ffmpeg(),
        check_whisper(),
        check_ollama(),
        check_permissions()
    ])
    if critical:
        print("✅ Environment ready! You can now run `vaio ./video.mp4` safely.")
        sys.exit(0)
    else:
        print("⚠️ Some checks failed. Fix above issues before running VAIO.")
        sys.exit(1)


def _check_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


# ────────────────────────────────
# CLI
# ────────────────────────────────
def main(argv=None):
    banner = dedent(
        f"""
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃   VAIO v{PROJECT_VERSION} – Video Auto Intelligence   ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        """
    )
    print(banner)

    # --- Auto-staged run: `vaio <video.ext>`
    if len(sys.argv) == 2 and Path(sys.argv[1]).suffix.lower() in (".mp4", ".mov", ".mkv"):
        video_path = Path(sys.argv[1]).resolve()
        print(f"🧩 Auto mode: Detected video file '{video_path.name}'")
        return full_auto.run(video_path)

    # --- Help when no args ---
    if len(sys.argv) == 1:
        print("Usage: vaio <command> <video.mp4> [options]\n")
        print("Commands:")
        print("  audio       🎧 Extract audio and generate captions")
        print("  desc        📝 Generate SEO title + description")
        print("  translate   🌐 Translate TD file into all languages")
        print("  captions    💬 Translate captions into all languages")
        print("  continue    ⏭️  Resume automatically from last stage")
        print("  check       🧪 Run environment diagnostics\n")
        print("Or simply:")
        print("  vaio ./myvideo.mp4   → auto-staged pipeline\n")
        sys.exit(0)

    # --- Regular CLI mode ---
    parser = argparse.ArgumentParser(
        prog="vaio",
        description="VAIO — Video Auto Intelligence Operator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"vaio {PROJECT_VERSION}")
    sub = parser.add_subparsers(dest="command")

    # add full-auto subcommand
    p_full = sub.add_parser("full-auto", help="🤖 Run full staged pipeline end-to-end")
    p_full.add_argument("video_file", help="Path to video file")
    p_full.add_argument("--template-file", help="Optional path to template.txt")
    p_full.set_defaults(func=lambda args: full_auto.run(
        Path(args.video_file).resolve(),
        Path(args.template_file).resolve() if args.template_file else None
    ))

    # commands
    p_audio = sub.add_parser("audio", help="🎧 Extract audio and generate captions")
    p_audio.add_argument("video_file")
    p_audio.set_defaults(func=handle_audio)

    p_desc = sub.add_parser("desc", help="📝 Generate SEO title + description (TD)")
    p_desc.add_argument("video_file")
    p_desc.add_argument("--template-file")
    p_desc.set_defaults(func=handle_desc)

    p_tr = sub.add_parser("translate", help="🌐 Translate TD file into all languages")
    p_tr.add_argument("video_file")
    p_tr.set_defaults(func=handle_translate)

    p_capt = sub.add_parser("captions", help="💬 Translate captions into all languages")
    p_capt.add_argument("video_file")
    p_capt.set_defaults(func=handle_captions)

    p_check = sub.add_parser("check", help="🧪 Run environment diagnostics")
    p_check.set_defaults(func=handle_debug)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
