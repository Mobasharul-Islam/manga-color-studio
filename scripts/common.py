"""Helpers shared by the server, the notebook builder and the Kaggle scripts."""
import os
import re
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Private Kaggle dataset that carries the scans of the chapter being colorized.
SCANS_DATASET_SLUG = "manga-color-scans"
KERNEL_SLUG = "tpu-image-generation-anima-base-1-0-2b"


def natural_key(name: str):
    """Sort key that orders 'page_2' before 'page_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def safe_name(name: str, fallback: str = "chapter_01") -> str:
    return re.sub(r"[^\w\-.]", "_", name or "") or fallback


def resolve_kaggle_username(api=None) -> str:
    """Returns the configured Kaggle username, or raises if none is configured."""
    if api is not None:
        cfg = getattr(api, "config_values", None) or {}
        if cfg.get("username"):
            return cfg["username"]
        try:
            val = api.get_config_value("username")
            if val:
                return val
        except Exception:
            pass

    u = os.getenv("KAGGLE_USERNAME")
    if u and u != "your_kaggle_username":
        return u

    try:
        kjson = Path.home() / ".kaggle" / "kaggle.json"
        if kjson.exists():
            data = json.loads(kjson.read_text(encoding="utf-8"))
            if data.get("username"):
                return data["username"]
    except Exception:
        pass

    raise RuntimeError("Kaggle username is not configured. Open 'API Tokens' and connect your Kaggle account.")
