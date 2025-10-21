from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from .query import build_kb_for_video, set_kb_dir_for_video, _resolve_kb_dir_for_video
from .paths import DEFAULT_KB_DIR, ensure_default_dirs
from .store import collection_stats, clear_index

def register_kb_cli(subparsers):
    kb = subparsers.add_parser("kb", help="Knowledge Base tools")
    kb_sub = kb.add_subparsers(dest="kb_cmd")

    p_build = kb_sub.add_parser("build", help="Build KB for a project video")
    p_build.add_argument("video", type=str, help="Path to project video")
    p_build.add_argument("--knowledge", type=str, default=None, help="Override knowledge directory (full path)")

    p_set = kb_sub.add_parser("set", help="Set knowledge dir (or disable) for a project video")
    p_set.add_argument("video", type=str, help="Path to project video")
    p_set.add_argument("--knowledge", type=str, default=None, help="Full path to knowledge dir; use 'none' to disable")

    p_stats = kb_sub.add_parser("stats", help="Show KB stats for a project video")
    p_stats.add_argument("video", type=str, help="Path to project video")

    p_clear = kb_sub.add_parser("clear", help="Clear KB index for a project video (keeps files)")
    p_clear.add_argument("video", type=str, help="Path to project video")

def handle_kb(args):
    ensure_default_dirs()
    cmd = args.kb_cmd
    if cmd == "build":
        video = Path(args.video)
        kb_dir = None if args.knowledge is None else Path(args.knowledge)
        if kb_dir is None:
            print(f"ℹ️  No --knowledge given; using project config or default.")
        result = build_kb_for_video(video, kb_dir)
        print(f"✅ KB {result['status']} | dir={result['kb']} | docs={result['count']}")
    elif cmd == "set":
        video = Path(args.video)
        if args.knowledge is None:
            print("❌ --knowledge is required (path or 'none')")
            return
        if args.knowledge.strip().lower() in {"none", "null"}:
            set_kb_dir_for_video(video, None)
            print("✅ KB disabled for this project.")
        else:
            kb_dir = Path(args.knowledge)
            kb_dir.mkdir(parents=True, exist_ok=True)
            set_kb_dir_for_video(video, kb_dir)
            print(f"✅ KB directory set to {kb_dir}")
    elif cmd == "stats":
        kb_dir = _resolve_kb_dir_for_video(Path(args.video))
        if kb_dir is None:
            print("ℹ️  KB disabled (knowledge=null).")
            return
        stats = collection_stats(kb_dir)
        print(f"📊 KB collection={stats['collection']} | docs={stats['count']} | dir={kb_dir}")
    elif args.kb_cmd == "clear":
        from .store import clear_index
        kb_dir = _resolve_kb_dir_for_video(Path(args.video))
        clear_index(kb_dir)
        return
