"""
utils/__init__.py
==================
Utility package for EmotionCine.

Exposes path resolution and output management helpers.
"""
from utils.path_resolver import (
    resolve_legacy_path,
    get_bundled_keyframes_dir,
    get_sample_outputs_dir,
    get_outputs_root,
    safe_read_json,
    safe_read_html,
)
from utils.output_manager import (
    ensure_outputs_dirs,
    save_summary_json,
    save_scene_metadata_json,
    save_emotion_csv,
    save_evaluation_json,
)

__all__ = [
    "resolve_legacy_path", "get_bundled_keyframes_dir",
    "get_sample_outputs_dir", "get_outputs_root",
    "safe_read_json", "safe_read_html",
    "ensure_outputs_dirs", "save_summary_json",
    "save_scene_metadata_json", "save_emotion_csv", "save_evaluation_json",
]
