"""Helpers shared by the server, the notebook builder and the Kaggle scripts."""
import os
import re
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Private Kaggle dataset that carries the scans of the chapter being colorized.
SCANS_DATASET_SLUG = "manga-color-scans"
# Encrypted runs use their own dataset: the plain one keeps unencrypted scans in its version history.
ENCRYPTED_SCANS_DATASET_SLUG = "manga-color-scans-enc"


# Prefix of the private one-run datasets that carry the encryption key to the notebook; the app
# deletes each one when its run ends.
KEY_DATASET_PREFIX = "mcs-key-"


def scans_dataset_slug(encrypted: bool) -> str:
    return ENCRYPTED_SCANS_DATASET_SLUG if encrypted else SCANS_DATASET_SLUG


KERNEL_SLUG = "tpu-image-generation-anima-base-1-0-2b"

# Hardware a colorization run can use. Each gets its own Kaggle notebook, so a run queued on
# one never blocks another (Kaggle's API can't cancel runs). Titles must slugify to the slug.
ACCELERATORS = {
    "tpu": {
        "label": "TPU v5e-8", "quota": "TPU", "backend": "tpu", "slug": KERNEL_SLUG,
        "title": "TPU Image Generation Anima Base 1.0 2B", "dir": "kernel_anima", "machine_shape": None,
    },
    "gpu_t4": {
        "label": "GPU T4 x2", "quota": "GPU", "backend": "gpu", "slug": "manga-color-gpu-t4",
        "title": "Manga Color GPU T4", "dir": "kernel_gpu_t4", "machine_shape": "NvidiaTeslaT4",
    },
    "gpu_p100": {
        "label": "GPU P100", "quota": "GPU", "backend": "gpu", "slug": "manga-color-gpu-p100",
        "title": "Manga Color GPU P100", "dir": "kernel_gpu_p100", "machine_shape": "NvidiaTeslaP100",
    },
}


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
