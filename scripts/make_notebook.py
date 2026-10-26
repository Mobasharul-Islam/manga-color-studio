"""Builds the Kaggle notebook (+ kernel-metadata.json) for one colorization run.

The notebook is a thin wrapper: install packages, check the environment, write
scripts/colorize_worker.py with this run's settings prepended as CONFIG, and call its main().
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Root paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from scripts.common import ACCELERATORS, SCANS_DATASET_SLUG, resolve_kaggle_username

WORKER_FILE = BASE_DIR / "scripts" / "colorize_worker.py"
DEFAULT_PROMPT = "masterpiece, best quality, anime style, highly detailed coloring, vibrant color harmony, crisp lineart"
DEFAULT_NEGATIVE = "worst quality, low quality, blurry, artifact, oversaturated, deformed, greyscale, monochrome"

INSTALL_CELL = [
    "!pip uninstall -y -q google-tunix 2>/dev/null; pip install -q --upgrade \"transformers==5.15.0\" \"tokenizers==0.22.2\" accelerate safetensors peft kaggle\n",
    "!pip install --upgrade \"huggingface_hub @ git+https://github.com/huggingface/huggingface_hub.git@f34cd32cac1f3603e3294543e1fb0eb4508d5139\" \"diffusers @ git+https://github.com/huggingface/diffusers.git@102c24934fe1705e6caaadb03789b0f2542fd8a9\"\n",
]

ENV_CHECK = """import os

import huggingface_hub
try:
    from huggingface_hub import get_cached_repo_tree  # noqa: F401
except ImportError:
    raise RuntimeError('Installed huggingface_hub is too old for diffusers main (' + huggingface_hub.__version__
                       + '); re-run the install cell or restart the session and Run All.')
import torch
import diffusers
import transformers

print('huggingface_hub:', huggingface_hub.__version__)
print('torch:', torch.__version__, '| diffusers:', diffusers.__version__, '| transformers:', transformers.__version__)
"""

TPU_ENV_CHECK = """# Remove Kaggle single-host env vars to unlock 8-core distributed multiprocessing
os.environ.pop('TPU_PROCESS_ADDRESSES', None)
os.environ.pop('CLOUD_TPU_TASK_ID', None)
import torch_xla
print('torch_xla:', torch_xla.__version__, '| PJRT_DEVICE:', os.getenv('PJRT_DEVICE'))
"""

GPU_ENV_CHECK = """if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU in this session. Check the notebook accelerator setting.')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print('GPU %d: %s, %.1f GB, compute %d.%d' % (i, p.name, p.total_memory / 1e9, p.major, p.minor))
"""


def _code_cell(source: str) -> Dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in source.splitlines()]}


def build_notebook(
    input_dir: Optional[str] = None,
    loras: Optional[List[Dict[str, Any]]] = None,
    model: Optional[Dict[str, Any]] = None,
    max_dim: int = 1440,
    num_steps: int = 8,
    guidance_scale: float = 1.0,
    prompt: str = "",
    negative_prompt: str = "",
    accelerator: str = "tpu",
    out_notebook_path: Optional[str] = None,
):
    acc = ACCELERATORS[accelerator]
    max_dim = min(int(max_dim), 1440)

    # The app passes the model & LoRAs selected in its library; standalone runs use the defaults.
    if model is None:
        model = {"name": "Anima Base 1.0 (2B)", "source": "circlestone-labs/Anima-Base-v1.0-Diffusers", "revision": None, "dataset": None}
    if loras is None:
        loras = [
            {"name": "Color Manga Anima", "path": "Kutches/Anim4/colorMangaAnima.safetensors", "scale": 1.0, "enabled": True, "dataset": None},
            {"name": "Anima Turbo", "path": "circlestone-labs/Anima-Official-LoRAs/anima-turbo-lora-v0.2.safetensors", "scale": 1.0, "enabled": True, "dataset": None},
        ]
    loras = [l for l in loras if l.get("enabled", True) and l.get("path")]

    # Only the username is embedded in the notebook; no keys or tokens.
    api = None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception:
        api = None
    kuser = resolve_kaggle_username(api)

    # Mount the scans, the selected model's dataset and the datasets of the enabled LoRAs.
    dataset_sources = [f"{kuser}/{SCANS_DATASET_SLUG}"]
    for ref in [model.get("dataset")] + [l.get("dataset") for l in loras]:
        if ref and ref not in dataset_sources:
            dataset_sources.append(ref)

    config = {
        "backend": acc["backend"],
        "model_repo": model["source"],
        "model_revision": model.get("revision") or "main",
        "loras": [{"name": l.get("name", ""), "path": l["path"], "scale": float(l.get("scale", 1.0))} for l in loras],
        "max_dim": max_dim,
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "prompt": (prompt or "").strip() or DEFAULT_PROMPT,
        "negative_prompt": (negative_prompt or "").strip() or DEFAULT_NEGATIVE,
    }
    # JSON parsed at runtime: the settings never have to be valid Python literals.
    worker_source = f"import json\nCONFIG = json.loads({json.dumps(config)!r})\n\n" + WORKER_FILE.read_text(encoding="utf-8")
    run_cell = (
        "import sys\n"
        f"WORKER_SOURCE = {worker_source!r}\n"
        "with open('/kaggle/working/colorize_worker.py', 'w', encoding='utf-8') as f:\n"
        "    f.write(WORKER_SOURCE)\n"
        "if '/kaggle/working' not in sys.path:\n"
        "    sys.path.insert(0, '/kaggle/working')\n"
        "import colorize_worker\n"
        "colorize_worker.main()\n"
    )

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Manga Colorization (%s)\n" % acc["label"],
            "\n",
            "Generated by Manga Color Studio. The worker code is `scripts/colorize_worker.py` in the app;\n",
            "this run's settings are in its `CONFIG` line.\n",
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": INSTALL_CELL},
        _code_cell(ENV_CHECK + (TPU_ENV_CHECK if acc["backend"] == "tpu" else GPU_ENV_CHECK)),
        _code_cell(run_cell),
    ]

    is_tpu = acc["backend"] == "tpu"
    nb = {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "tpuV3-8" if is_tpu else acc["machine_shape"][0].lower() + acc["machine_shape"][1:],
                "dataSources": [{"datasetId": ref, "sourceType": "dataset"} for ref in dataset_sources],
                "dockerImageVersionId": 30787,
                "isGpuEnabled": not is_tpu,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_file = BASE_DIR / (out_notebook_path or f"{acc['dir']}/colorize.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    kernel_meta = {
        "id": f"{kuser}/{acc['slug']}",
        "title": acc["title"],
        "code_file": out_file.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false" if is_tpu else "true",
        "enable_tpu": "true" if is_tpu else "false",
        "enable_internet": "true",
        "dataset_sources": dataset_sources,
        "kernel_sources": [],
    }
    if acc["machine_shape"]:
        kernel_meta["machine_shape"] = acc["machine_shape"]
    (out_file.parent / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=1), encoding="utf-8")

    print("Successfully generated %s and kernel-metadata.json for @%s (%s)" % (out_file, kuser, acc["label"]))
    return str(out_file)


if __name__ == "__main__":
    build_notebook(input_dir=str(BASE_DIR / "input_pages"), accelerator=sys.argv[1] if len(sys.argv) > 1 else "tpu")
