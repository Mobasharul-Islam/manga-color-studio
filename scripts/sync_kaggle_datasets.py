"""
Script to create and synchronize Kaggle datasets:
1. Models_Manga_Coloring (mobasharulislam/models-manga-coloring)
2. Loras_Manga_Coloring_Anima (mobasharulislam/loras-manga-coloring-anima)
"""
import os
import sys
import glob
import json
import time
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

LORAS_DIR = BASE_DIR / "loras"
DATASETS_STAGING_DIR = BASE_DIR / "kaggle_datasets"
LORAS_STAGING_DIR = DATASETS_STAGING_DIR / "Loras_Manga_Coloring_Anima"
MODELS_STAGING_DIR = DATASETS_STAGING_DIR / "Models_Manga_Coloring"
LORAS_CONFIG = BASE_DIR / "loras.json"

LORAS_DIR.mkdir(parents=True, exist_ok=True)
LORAS_STAGING_DIR.mkdir(parents=True, exist_ok=True)
MODELS_STAGING_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def get_kaggle_api():
    """Initializes and authenticates Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def check_dataset_status(username: str = "mobasharulislam"):
    """Checks whether the datasets exist on Kaggle and returns their status."""
    api = get_kaggle_api()
    status = {
        "models": {
            "dataset_id": f"{username}/models-manga-coloring",
            "title": "Models_Manga_Coloring",
            "exists": False,
            "info": None
        },
        "loras": {
            "dataset_id": f"{username}/loras-manga-coloring-anima",
            "title": "Loras_Manga_Coloring_Anima",
            "exists": False,
            "info": None
        }
    }

    try:
        datasets = api.dataset_list(user=username)
        for ds in datasets:
            ref = getattr(ds, "ref", "") or getattr(ds, "id", "")
            title = getattr(ds, "title", "")
            if "models-manga-coloring" in ref.lower() or "models_manga_coloring" in title.lower():
                status["models"]["exists"] = True
                status["models"]["info"] = {
                    "ref": ref,
                    "title": title,
                    "url": getattr(ds, "url", f"https://www.kaggle.com/datasets/{ref}"),
                    "totalBytes": getattr(ds, "totalBytes", 0),
                    "version": getattr(ds, "currentVersionNumber", 1),
                    "lastUpdated": str(getattr(ds, "lastUpdated", ""))
                }
            if "loras-manga-coloring-anima" in ref.lower() or "loras_manga_coloring_anima" in title.lower():
                status["loras"]["exists"] = True
                status["loras"]["info"] = {
                    "ref": ref,
                    "title": title,
                    "url": getattr(ds, "url", f"https://www.kaggle.com/datasets/{ref}"),
                    "totalBytes": getattr(ds, "totalBytes", 0),
                    "version": getattr(ds, "currentVersionNumber", 1),
                    "lastUpdated": str(getattr(ds, "lastUpdated", ""))
                }
    except Exception as e:
        print(f"Error checking dataset status: {e}")

    return status


def prepare_local_loras():
    """Ensures built-in LoRAs and custom LoRAs are present in local loras/ folder."""
    # Check default HF cache
    hf_cache = Path(os.environ["HF_HOME"]) / "hub"
    
    # 1. Color Manga Anima
    target1 = LORAS_DIR / "colorMangaAnima.safetensors"
    if not target1.exists():
        found = list(hf_cache.glob("**/colorMangaAnima.safetensors"))
        if found and found[0].exists():
            shutil.copy2(found[0], target1)
            print(f"Copied {target1.name} from local HF cache.")
        else:
            print("Downloading colorMangaAnima.safetensors from HuggingFace...")
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id="Kutches/Anim4", filename="colorMangaAnima.safetensors")
            shutil.copy2(p, target1)

    # 2. Anima Turbo LoRA
    target2 = LORAS_DIR / "anima-turbo-lora-v0.2.safetensors"
    if not target2.exists():
        found = list(hf_cache.glob("**/anima-turbo-lora-v0.2.safetensors"))
        if found and found[0].exists():
            shutil.copy2(found[0], target2)
            print(f"Copied {target2.name} from local HF cache.")
        else:
            print("Downloading anima-turbo-lora-v0.2.safetensors from HuggingFace...")
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id="circlestone-labs/Anima-Official-LoRAs", filename="anima-turbo-lora-v0.2.safetensors")
            shutil.copy2(p, target2)

    # Bidirectional sync between loras/ and staging directory
    for f in LORAS_STAGING_DIR.glob("*.safetensors"):
        dst = LORAS_DIR / f.name
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dst)

    for f in LORAS_DIR.glob("*.safetensors"):
        dst = LORAS_STAGING_DIR / f.name
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dst)

    lora_files = list(LORAS_STAGING_DIR.glob("*.safetensors"))
    print(f"Staged {len(lora_files)} LoRAs in {LORAS_STAGING_DIR}: {[f.name for f in lora_files]}")
    return lora_files


def sync_loras_dataset(username: str = "mobasharulislam"):
    """Syncs the Loras_Manga_Coloring_Anima dataset to Kaggle."""
    api = get_kaggle_api()
    prepare_local_loras()

    dataset_slug = "loras-manga-coloring-anima"
    dataset_ref = f"{username}/{dataset_slug}"

    meta = {
        "title": "Loras_Manga_Coloring_Anima",
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(LORAS_STAGING_DIR / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    status = check_dataset_status(username)
    if status["loras"]["exists"]:
        print(f"Updating existing Kaggle dataset '{dataset_ref}'...")
        api.dataset_create_version(
            str(LORAS_STAGING_DIR),
            version_notes=f"Updated LoRAs ({time.strftime('%Y-%m-%d %H:%M:%S')})",
            dir_mode="zip"
        )
        print(f"Successfully updated Kaggle dataset '{dataset_ref}'!")
    else:
        print(f"Creating new Kaggle dataset '{dataset_ref}'...")
        api.dataset_create_new(
            str(LORAS_STAGING_DIR),
            dir_mode="zip",
            public=False,
            quiet=False
        )
        print(f"Successfully created Kaggle dataset '{dataset_ref}'!")


def prepare_local_models():
    """Downloads/copies the full Anima Base 1.0 Diffusers model to staging folder."""
    from huggingface_hub import snapshot_download
    print("Downloading/Verifying Anima-Base-v1.0-Diffusers model files...")
    snapshot_dir = snapshot_download(
        repo_id="circlestone-labs/Anima-Base-v1.0-Diffusers",
        token=HF_TOKEN,
        local_dir=str(MODELS_STAGING_DIR),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git*", "README.md", "LICENSE.md"]
    )
    print(f"Model staged successfully in {MODELS_STAGING_DIR}")
    return snapshot_dir


def sync_models_dataset(username: str = "mobasharulislam"):
    """Syncs the Models_Manga_Coloring dataset to Kaggle."""
    api = get_kaggle_api()
    prepare_local_models()

    dataset_slug = "models-manga-coloring"
    dataset_ref = f"{username}/{dataset_slug}"

    meta = {
        "title": "Models_Manga_Coloring",
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(MODELS_STAGING_DIR / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    status = check_dataset_status(username)
    if status["models"]["exists"]:
        print(f"Updating existing Kaggle dataset '{dataset_ref}'...")
        api.dataset_create_version(
            str(MODELS_STAGING_DIR),
            version_notes=f"Anima Base 1.0 Diffusers Model Snapshot ({time.strftime('%Y-%m-%d %H:%M:%S')})",
            dir_mode="zip"
        )
        print(f"Successfully updated Kaggle dataset '{dataset_ref}'!")
    else:
        print(f"Creating new Kaggle dataset '{dataset_ref}'...")
        api.dataset_create_new(
            str(MODELS_STAGING_DIR),
            dir_mode="zip",
            public=False,
            quiet=False
        )
        print(f"Successfully created Kaggle dataset '{dataset_ref}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kaggle Datasets for Manga Colorization TPU Studio")
    parser.add_argument("--check", action="store_true", help="Check status of datasets on Kaggle")
    parser.add_argument("--sync-loras", action="store_true", help="Sync LoRAs dataset to Kaggle")
    parser.add_argument("--sync-models", action="store_true", help="Sync Anima Base model dataset to Kaggle")
    parser.add_argument("--sync-all", action="store_true", help="Sync both LoRAs and Models datasets")
    args = parser.parse_args()

    if args.check:
        st = check_dataset_status()
        print(json.dumps(st, indent=2))
    elif args.sync_loras:
        sync_loras_dataset()
    elif args.sync_models:
        sync_models_dataset()
    elif args.sync_all:
        sync_loras_dataset()
        sync_models_dataset()
    else:
        # Default action if no flags
        st = check_dataset_status()
        print(json.dumps(st, indent=2))
