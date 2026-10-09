"""Model & LoRA library.

Every model and every LoRA can be cached in its own private Kaggle dataset
(`<user>/mcs-model-<slug>`, `<user>/mcs-lora-<slug>`), each holding a `library.json`
that describes it. One dataset per item means adding or deleting never re-uploads
anything else. TPU runs mount only the selected model and the enabled LoRAs.

Caching from a link runs in a small CPU-only Kaggle notebook (scripts/library_worker.py),
so it costs no TPU quota and no TPU queue time. LoRAs uploaded from the PC (or downloaded
locally with a Civitai key) are published straight from this machine.

State lives in library.json next to the app; Kaggle is the source of truth for what is cached.
"""
import json
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from scripts.common import resolve_kaggle_username, safe_name

BASE_DIR = Path(__file__).resolve().parent.parent
LIBRARY_FILE = BASE_DIR / "library.json"
LEGACY_LORAS_FILE = BASE_DIR / "loras.json"
LORAS_DIR = BASE_DIR / "loras"
WORKER_DIR = BASE_DIR / "kernel_library"
WORKER_SCRIPT = BASE_DIR / "scripts" / "library_worker.py"
WORKER_SLUG = "manga-color-library-worker"
WORKER_TIMEOUT_S = 2 * 3600

# Datasets created by earlier versions of the app. They are recognised so their contents
# keep being used, and listed so they can be deleted once no longer needed.
LEGACY_MODEL_SLUG = "models-manga-coloring"
LEGACY_LORA_SLUG = "loras-manga-coloring-anima"
LEGACY_SLUGS = {
    LEGACY_MODEL_SLUG: "Anima Base model (old layout, not used for loading before this version)",
    LEGACY_LORA_SLUG: "Shared LoRA dataset (also contains a stray copy of the text encoder & VAE)",
    "gbd-ch-107-part2": "Old scans dataset (replaced by manga-color-scans)",
}

ANIMA_BASE = {
    "id": "anima_base",
    "kind": "model",
    "name": "Anima Base 1.0 (2B)",
    "source": "circlestone-labs/Anima-Base-v1.0-Diffusers",
    "revision": None,
    "description": "Official Anima Base Diffusers weights.",
}

DEFAULT_LORAS = [
    {
        "id": "color_manga",
        "name": "Color Manga Anima (Base Style)",
        "source": "Kutches/Anim4/colorMangaAnima.safetensors",
        "filename": "colorMangaAnima.safetensors",
        "scale": 1.0,
        "enabled": True,
        "is_turbo": False,
        "description": "Transforms lineart into rich, full-color anime shading.",
    },
    {
        "id": "turbo_8step",
        "name": "Anima Turbo (8-Step Fast)",
        "source": "circlestone-labs/Anima-Official-LoRAs/anima-turbo-lora-v0.2.safetensors",
        "filename": "anima-turbo-lora-v0.2.safetensors",
        "scale": 1.0,
        "enabled": True,
        "is_turbo": True,
        "description": "Accelerates generation down to 8 steps per page on TPU.",
    },
]

ITEM_DEFAULTS = {
    "kind": "lora",
    "source": "",
    "revision": None,
    "filename": None,
    "dataset": None,        # owner/slug of the Kaggle dataset holding this item
    "legacy": False,        # dataset is a shared legacy dataset: never deleted with the item
    "status": "not_cached", # cached | not_cached | caching | failed
    "error": None,
    "size_bytes": None,
    "scale": 1.0,
    "enabled": False,
    "is_turbo": False,
    "description": "",
}

_log: Callable[[str], None] = print
_get_api: Callable[[], Any] = lambda: None
_lock = threading.RLock()
_tasks: "queue.Queue[tuple]" = queue.Queue()
_task_state: Dict[str, Any] = {"current": None, "pending": 0, "history": []}
_worker_thread: Optional[threading.Thread] = None
_data: Dict[str, Any] = {}

router = APIRouter(prefix="/api")


def init(log_fn: Callable[[str], None], api_fn: Callable[[], Any]):
    global _log, _get_api, _data
    _log, _get_api = log_fn, api_fn
    with _lock:
        _data = _load()
        # A 'caching' status left over from a previous server run will never finish.
        for item in _all_items():
            if item["status"] == "caching":
                item["status"] = "failed"
                item["error"] = "Interrupted: the app was restarted while this was caching. Retry it."
        _save()


# ---------------------------------------------------------------- registry

def _item(**fields) -> Dict[str, Any]:
    item = dict(ITEM_DEFAULTS)
    item.update({k: v for k, v in fields.items() if v is not None or k in ("dataset", "error")})
    return item


def _load() -> Dict[str, Any]:
    if LIBRARY_FILE.exists():
        try:
            data = json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
            data["models"] = [_item(**m) for m in data.get("models", [])]
            data["loras"] = [_item(**l) for l in data.get("loras", [])]
            data.setdefault("legacy", [])
            data.setdefault("active_model", "anima_base")
            return data
        except Exception as e:
            print("Error reading library.json, rebuilding it:", e)

    # First run: seed from the old loras.json so existing LoRAs and their weights carry over.
    loras = [_item(kind="lora", **l) for l in DEFAULT_LORAS]
    known = {l["filename"] for l in loras}
    try:
        old = json.loads(LEGACY_LORAS_FILE.read_text(encoding="utf-8")).get("loras", [])
    except Exception:
        old = []
    for o in old:
        fname = o.get("filename") or Path(str(o.get("path", ""))).name
        match = next((l for l in loras if l["filename"] == fname), None)
        if match:
            match["scale"] = float(o.get("scale", match["scale"]))
            match["enabled"] = bool(o.get("enabled", match["enabled"]))
            continue
        if not fname or fname in known:
            continue
        local_only = not ("/" in str(o.get("path", "")) or str(o.get("path", "")).startswith("http"))
        loras.append(_item(
            kind="lora", id=o.get("id") or _slug(fname), name=o.get("name") or fname,
            source=fname if local_only else o.get("path"), filename=fname,
            scale=float(o.get("scale", 1.0)), enabled=bool(o.get("enabled", False)),
            is_turbo=bool(o.get("is_turbo", False)), description=o.get("description", ""),
        ))
        known.add(fname)
    return {"active_model": "anima_base", "models": [_item(**ANIMA_BASE)], "loras": loras, "legacy": []}


def _save():
    tmp = LIBRARY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_data, indent=2), encoding="utf-8")
    tmp.replace(LIBRARY_FILE)


def _all_items() -> List[Dict[str, Any]]:
    return _data["models"] + _data["loras"]


def _find(item_id: str) -> Optional[Dict[str, Any]]:
    return next((i for i in _all_items() if i["id"] == item_id), None)


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", Path(text).stem.lower())).strip("-") or "item"


def _unique_id(base: str) -> str:
    base = re.sub(r"[^a-z0-9_]", "_", base.lower()).strip("_") or "item"
    ids = {i["id"] for i in _all_items()}
    candidate, n = base, 2
    while candidate in ids:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def _dataset_ref_for(item: Dict[str, Any], owner: str) -> str:
    """Stable, unique dataset slug for an item: mcs-<kind>-<name>, at most 50 characters."""
    taken = {(i.get("dataset") or "").lower() for i in _all_items() if i is not item}
    base = f"mcs-{item['kind']}-{_slug(item['name'])}"
    if len(base) > 44:  # Kaggle slugs max out at 50; leave room for a -N suffix
        base = base[:45].rsplit("-", 1)[0]
    slug, n = base, 2
    while f"{owner}/{slug}".lower() in taken:
        slug, n = f"{base}-{n}", n + 1
    return f"{owner}/{slug}"


def is_usable(item: Dict[str, Any]) -> bool:
    """Whether a TPU run can use the item: cached on Kaggle, or downloadable from Hugging Face."""
    if item["status"] == "cached" and item.get("dataset"):
        return True
    src = item.get("source") or ""
    if item["kind"] == "model":
        return bool(src)
    return src.startswith("https://huggingface.co/") or (not src.startswith("http") and src.count("/") >= 2)


def snapshot() -> Dict[str, Any]:
    with _lock:
        data = copy.deepcopy(_data)
        tasks = copy.deepcopy(_task_state)
    for item in data["models"] + data["loras"]:
        item["usable"] = is_usable(item)
    data["tasks"] = tasks
    data["worker_notebook_url"] = _worker_url()
    return data


def _worker_url() -> Optional[str]:
    try:
        return f"https://www.kaggle.com/code/{resolve_kaggle_username(_get_api())}/{WORKER_SLUG}"
    except Exception:
        return None


# ---------------------------------------------------------------- run spec for TPU jobs

def run_spec() -> Dict[str, Any]:
    """Model + LoRAs for a TPU run. Raises ValueError when something selected can't be used."""
    with _lock:
        model = _find(_data.get("active_model")) or _find("anima_base")
        if model is None or not is_usable(model):
            raise ValueError("Select a model in the library before colorizing.")
        loras = []
        for l in _data["loras"]:
            if not l["enabled"]:
                continue
            if not is_usable(l):
                raise ValueError(f"LoRA '{l['name']}' is enabled but not cached on Kaggle yet. Cache it or disable it.")
            cached = l["status"] == "cached" and l.get("dataset")
            loras.append({
                "name": l["name"],
                # A cached LoRA is found by file name in the mounted dataset; otherwise it is
                # downloaded from its Hugging Face source inside the notebook.
                "path": l["filename"] if cached else l["source"],
                "filename": l["filename"],
                "scale": float(l["scale"]),
                "enabled": True,
                "dataset": l["dataset"] if cached else None,
            })
        return {
            "model": {
                "name": model["name"],
                "source": model["source"],
                "revision": model.get("revision"),
                "dataset": model["dataset"] if model["status"] == "cached" else None,
            },
            "loras": loras,
        }


# ---------------------------------------------------------------- Kaggle sync

def _list_my_datasets(api) -> Dict[str, Any]:
    found = {}
    for page in range(1, 20):
        batch = [d for d in (api.dataset_list(mine=True, page=page) or []) if d]
        for d in batch:
            found[d.ref.lower()] = d
        if len(batch) < 20:
            break
    return found


def refresh() -> Dict[str, Any]:
    """Re-reads which items are cached from the user's Kaggle datasets."""
    api = _get_api()
    if api is None:
        raise RuntimeError("Kaggle is not connected.")
    owner = resolve_kaggle_username(api)
    datasets = _list_my_datasets(api)

    legacy_lora_files = set()
    legacy_lora_ref = f"{owner}/{LEGACY_LORA_SLUG}".lower()
    if legacy_lora_ref in datasets:
        try:
            legacy_lora_files = {f.name for f in api.dataset_list_files(legacy_lora_ref).files}
        except Exception as e:
            _log(f"Library: could not list {legacy_lora_ref}: {e}")

    # library.json of datasets this machine doesn't know yet (e.g. cached from another PC)
    with _lock:
        known_refs = {(i.get("dataset") or "").lower() for i in _all_items()}
    discovered = []
    for ref, d in datasets.items():
        slug = ref.split("/", 1)[1]
        if not slug.startswith(("mcs-model-", "mcs-lora-")) or ref in known_refs:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                api.dataset_download_file(d.ref, "library.json", path=tmp, quiet=True)
                entry = json.loads(next(Path(tmp).rglob("library.json")).read_text(encoding="utf-8"))
            discovered.append((d, entry))
        except Exception as e:
            _log(f"Library: skipping {d.ref} (no readable library.json: {e})")

    with _lock:
        for d, entry in discovered:
            kind = "model" if d.ref.lower().split("/", 1)[1].startswith("mcs-model-") else "lora"
            items = _data["models" if kind == "model" else "loras"]
            # Re-checked under the lock: a concurrent refresh may have added it already.
            if any((i.get("dataset") or "").lower() == d.ref.lower() for i in items):
                continue
            # The same model / LoRA may already be listed without its dataset (e.g. a fresh
            # library.json on this PC): attach the dataset instead of adding a duplicate.
            same = next((i for i in items if (i.get("dataset") is None or i.get("legacy")) and (
                (kind == "lora" and entry.get("filename") and i.get("filename") == entry.get("filename"))
                or (kind == "model" and i.get("source", "").lower() == (entry.get("source") or "").lower()
                    and i.get("revision") == entry.get("revision")))), None)
            if same is not None:
                same.update(dataset=d.ref, legacy=False)
                continue
            items.append(_item(
                kind=kind, id=_unique_id(entry.get("id") or entry.get("name") or kind),
                name=entry.get("name") or d.title, source=entry.get("source", ""),
                revision=entry.get("revision"), filename=entry.get("filename"),
                description=entry.get("description", ""), scale=float(entry.get("scale", 1.0)),
                is_turbo=bool(entry.get("is_turbo", False)), dataset=d.ref,
            ))

        for item in _all_items():
            if item["status"] == "caching":
                continue
            ref = (item.get("dataset") or "").lower()
            if ref and ref in datasets and not item.get("legacy"):
                item.update(status="cached", error=None, size_bytes=datasets[ref].total_bytes)
                continue
            # Fall back to the datasets earlier versions of the app created.
            if item["id"] == "anima_base" and f"{owner}/{LEGACY_MODEL_SLUG}" in datasets:
                d = datasets[f"{owner}/{LEGACY_MODEL_SLUG}"]
                item.update(dataset=d.ref, legacy=True, status="cached", size_bytes=d.total_bytes)
            elif item["kind"] == "lora" and item.get("filename") in legacy_lora_files:
                item.update(dataset=datasets[legacy_lora_ref].ref, legacy=True, status="cached")
            else:
                if item["status"] == "cached":
                    item["status"] = "not_cached"
                if item.get("legacy") or (ref and ref not in datasets):
                    item.update(dataset=None, legacy=False)

        _data["legacy"] = [
            {"ref": datasets[f"{owner}/{slug}"].ref, "title": datasets[f"{owner}/{slug}"].title,
             "size_bytes": datasets[f"{owner}/{slug}"].total_bytes, "note": note,
             "in_use": any(i.get("legacy") and (i.get("dataset") or "").lower() == f"{owner}/{slug}" for i in _all_items())}
            for slug, note in LEGACY_SLUGS.items() if f"{owner}/{slug}" in datasets
        ]
        _save()
    return snapshot()


# ---------------------------------------------------------------- source parsing

def parse_model_source(text: str):
    """Returns (repo_id, revision) for a Hugging Face Diffusers repo, or raises ValueError."""
    text = (text or "").strip().rstrip("/")
    if "civitai" in text or text.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin")):
        raise ValueError(
            "Single-file checkpoints (for example from Civitai) aren't supported yet. "
            "Use a Hugging Face repo in Diffusers format, e.g. circlestone-labs/Anima-Base-v1.0-Diffusers."
        )
    m = re.match(r"https?://huggingface\.co/([\w.-]+)/([\w.-]+)(?:/tree/([^/?#]+))?/?$", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}", m.group(3)
    if re.fullmatch(r"[\w.-]+/[\w.-]+", text):
        return text, None
    raise ValueError("Enter a Hugging Face model repo, e.g. https://huggingface.co/org/model or org/model.")


def validate_model_repo(repo: str, revision: Optional[str]) -> Dict[str, Any]:
    """Checks on this machine, before any Kaggle time is spent, that the repo is an Anima model."""
    import os
    from huggingface_hub import HfApi, hf_hub_download

    token = os.getenv("HF_TOKEN") or None
    info = HfApi().model_info(repo, revision=revision, token=token, files_metadata=True)
    names = {s.rfilename: (s.size or 0) for s in info.siblings or []}
    index = next((n for n in ("modular_model_index.json", "model_index.json") if n in names), None)
    if index is None:
        raise ValueError(f"'{repo}' is not a Diffusers-format model (no model_index.json).")
    cfg = json.loads(Path(hf_hub_download(repo, index, revision=revision, token=token)).read_text(encoding="utf-8"))
    transformer = json.dumps(cfg.get("transformer", ""))
    if "Cosmos" not in transformer and "Anima" not in json.dumps(cfg.get("_class_name", "")):
        raise ValueError(
            f"'{repo}' does not look like an Anima model (transformer: {transformer[:80]}). "
            "The colorizer only supports Anima-based models."
        )
    return {"size_bytes": sum(names.values())}


def resolve_civitai(url: str, token: str = "") -> tuple:
    """Civitai model page/API URL -> (direct download URL, file name, model name)."""
    url = re.sub(r"https?://civitai\.(red|green|blue|org)/", "https://civitai.com/", url)
    parsed = urlparse(url)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    version_id = parse_qs(parsed.query).get("modelVersionId", [None])[0]
    m = re.search(r"/models/(\d+)", parsed.path)
    try:
        if m and version_id:
            v = requests.get(f"https://civitai.com/api/v1/model-versions/{version_id}", headers=headers, timeout=15).json()
            name = (v.get("model") or {}).get("name") or v.get("name", "")
        elif m:
            data = requests.get(f"https://civitai.com/api/v1/models/{m.group(1)}", headers=headers, timeout=15).json()
            v = (data.get("modelVersions") or [{}])[0]
            name = data.get("name", "")
        else:
            return url, "", ""
        files = v.get("files") or []
        primary = next((f for f in files if f.get("primary")), files[0] if files else {})
        dl = primary.get("downloadUrl") or v.get("downloadUrl") or f"https://civitai.com/api/download/models/{v.get('id')}"
        return dl, primary.get("name", ""), name
    except Exception as e:
        _log(f"Civitai lookup note: {e}")
        return url, "", ""


def parse_lora_source(text: str) -> Dict[str, str]:
    """Normalises a LoRA link. Returns source, download_url, filename and a suggested name."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter a LoRA link or Hugging Face path.")
    m = re.match(r"https?://huggingface\.co/([^/]+)/([^/]+)/(?:blob|resolve)/([^/]+)/(.+)", text)
    if m:
        fname = Path(unquote(m.group(4))).name
        return {"source": text, "download_url": "", "filename": fname, "name": Path(fname).stem}
    if not text.startswith(("http://", "https://")):
        parts = text.split("/")
        if len(parts) < 3 or not parts[-1].endswith(".safetensors"):
            raise ValueError("Hugging Face paths look like org/repo/file.safetensors.")
        return {"source": text, "download_url": "", "filename": parts[-1], "name": Path(parts[-1]).stem}
    if "civitai" in text:
        dl, fname, name = resolve_civitai(text)
        return {"source": text, "download_url": dl, "filename": fname, "name": name or Path(fname).stem}
    fname = Path(unquote(urlparse(text).path)).name
    return {"source": text, "download_url": text,
            "filename": fname if fname.endswith(".safetensors") else "", "name": Path(fname).stem}


# ---------------------------------------------------------------- task queue

def _enqueue(task: tuple):
    global _worker_thread
    with _lock:
        _task_state["pending"] += 1
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_task_loop, daemon=True)
            _worker_thread.start()
    _tasks.put(task)


def _task_loop():
    while True:
        task = _tasks.get()
        action, item_id = task[0], task[1]
        with _lock:
            _task_state["pending"] -= 1
            item = _find(item_id)
            label = f"{action} {item['name'] if item else item_id}"
            _task_state["current"] = {"label": label, "started": time.time()}
        ok, err = True, None
        try:
            {"cache": _do_cache, "upload": _do_upload, "delete": _do_delete,
             "delete_legacy": _do_delete_legacy}[action](*task[1:])
        except Exception as e:
            ok, err = False, str(e)
            _log(f"Library: {label} failed: {e}")
            with _lock:
                item = _find(item_id)
                if item is not None and action in ("cache", "upload"):
                    item.update(status="failed", error=err)
                    _save()
        with _lock:
            _task_state["current"] = None
            _task_state["history"] = ([{"label": label, "ok": ok, "error": err, "finished": time.time()}]
                                      + _task_state["history"])[:10]


def _owner_and_api():
    api = _get_api()
    if api is None:
        raise RuntimeError("Kaggle is not connected. Open 'API Tokens' first.")
    return resolve_kaggle_username(api), api


def _write_worker_notebook(task: Dict[str, Any], owner: str):
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    # JSON parsed at runtime: dumping it as Python source would turn null/true into NameErrors.
    source = f"import json\nTASK = json.loads({json.dumps(task)!r})\n\n" + WORKER_SCRIPT.read_text(encoding="utf-8")
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": [line + "\n" for line in source.splitlines()]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (WORKER_DIR / "library_worker.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (WORKER_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{WORKER_SLUG}",
        "title": "Manga Color Library Worker",
        "code_file": "library_worker.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "kernel_sources": [],
    }, indent=1), encoding="utf-8")


def _wait_for_kernel(ref: str, label: str):
    deadline = time.time() + WORKER_TIMEOUT_S
    failures, last = 0, None
    while time.time() < deadline:
        time.sleep(10)
        res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", ref],
                             capture_output=True, text=True, cwd=BASE_DIR)
        text = (res.stdout or "").strip()
        if res.returncode != 0 or not text:
            failures += 1
            if failures >= 12:
                raise RuntimeError("Lost contact with Kaggle while waiting for the library notebook.")
            continue
        failures = 0
        status = next((s for s in ("QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCEL") if s in text), text)
        if status != last:
            _log(f"Library: {label}: notebook {status.lower()}")
            last = status
        if status == "COMPLETE":
            return
        if status in ("ERROR", "CANCEL"):
            raise RuntimeError(f"The library notebook stopped with status {text}. Open it on Kaggle for the log.")
    raise RuntimeError("The library notebook did not finish within 2 hours.")


def _read_worker_result(owner: str) -> Dict[str, Any]:
    """result.json written by the worker; if it crashed before writing one, the error from its log."""
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "download_fixed.py"), tmp, owner, WORKER_SLUG],
                             capture_output=True, text=True, cwd=BASE_DIR)
        result_file = next(Path(tmp).rglob("result.json"), None)
        if result_file is not None:
            return json.loads(result_file.read_text(encoding="utf-8"))
        error = ""
        log_file = next(Path(tmp).glob("*.log"), None)
        if log_file is not None:
            try:
                entries = json.loads(log_file.read_text(encoding="utf-8", errors="replace"))
                found = re.findall(r"^(\w+(?:Error|Exception): .+)$", "".join(e.get("data", "") for e in entries), re.M)
                error = found[-1] if found else ""
            except Exception:
                pass
        return {"ok": False, "error": f"The library notebook crashed: {error or (res.stderr or '').strip()[-300:]}"}


def _wait_dataset_ready(api, ref: str, timeout: int = 900):
    waited = 0
    while waited < timeout:
        try:
            status = str(api.dataset_status(ref)).lower()
            if "ready" in status:
                return
            if "error" in status or "fail" in status:
                raise RuntimeError(f"Kaggle failed to process dataset {ref}: {status}")
        except RuntimeError:
            raise
        except Exception:
            pass
        time.sleep(10)
        waited += 10
    _log(f"Library: {ref} is still being processed by Kaggle; it will show as cached once ready.")


def _library_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("id", "kind", "name", "source", "revision", "description", "scale", "is_turbo")
    return {k: item.get(k) for k in keys}


def _do_cache(item_id: str, download_url: str = ""):
    owner, api = _owner_and_api()
    with _lock:
        item = _find(item_id)
        if item is None:
            return
        if not item.get("dataset") or item.get("legacy"):
            item.update(dataset=_dataset_ref_for(item, owner), legacy=False)
        item.update(status="caching", error=None)
        _save()
        task = {
            "item_id": item["id"],
            "kind": item["kind"],
            "source": item["source"],
            "revision": item.get("revision"),
            "download_url": download_url,
            "filename": item.get("filename") or "",
            "owner": owner,
            "dataset": item["dataset"],
            "title": f"MCS {item['kind'].title()} {item['name']}"[:50].ljust(6, "_"),
            "notebook_url": f"https://www.kaggle.com/code/{owner}/{WORKER_SLUG}/edit",
            "library": _library_entry(item),
        }
        label = item["name"]

    _log(f"Library: caching '{label}' via a CPU notebook (no TPU time used)...")
    _write_worker_notebook(task, owner)
    push = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(WORKER_DIR)],
                          capture_output=True, text=True, cwd=BASE_DIR)
    if push.returncode != 0:
        raise RuntimeError(f"Could not push the library notebook: {(push.stderr or push.stdout).strip()[-300:]}")
    try:
        _wait_for_kernel(f"{owner}/{WORKER_SLUG}", label)
    except RuntimeError as e:
        if "stopped with status" not in str(e):
            raise  # timeouts / lost contact; a crashed notebook still has a log to read
    result = _read_worker_result(owner)
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "Unknown error in the library notebook.")
    _wait_dataset_ready(api, task["dataset"])
    with _lock:
        item = _find(item_id)
        if item is not None:
            item.update(status="cached", error=None, size_bytes=result.get("size_bytes"),
                        filename=result.get("filename") or item.get("filename"))
            _save()
    _log(f"Library: '{label}' is cached in {task['dataset']}.")


def _do_upload(item_id: str, local_path: str):
    """Publishes a LoRA file from this PC as its own dataset (small, no notebook needed)."""
    owner, api = _owner_and_api()
    with _lock:
        item = _find(item_id)
        if item is None:
            return
        if not item.get("dataset") or item.get("legacy"):
            item.update(dataset=_dataset_ref_for(item, owner), legacy=False)
        item.update(status="caching", error=None)
        _save()
        ref, label, entry = item["dataset"], item["name"], _library_entry(item)

    src = Path(local_path)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy2(src, stage / src.name)
        entry.update(filename=src.name, size_bytes=src.stat().st_size, added=time.strftime("%Y-%m-%d %H:%M:%S"))
        (stage / "library.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        (stage / "dataset-metadata.json").write_text(json.dumps({
            "title": f"MCS Lora {label}"[:50].ljust(6, "_"), "id": ref, "licenses": [{"name": "unknown"}],
        }), encoding="utf-8")
        _log(f"Library: uploading '{label}' ({src.stat().st_size / 1e6:.0f} MB) to {ref}...")
        exists = ref.lower() in {d.ref.lower() for d in (api.dataset_list(mine=True, search=ref.split('/')[1]) or []) if d}
        if exists:
            res = api.dataset_create_version(str(stage), version_notes="Re-uploaded from PC", dir_mode="skip")
        else:
            res = api.dataset_create_new(str(stage), public=False, dir_mode="skip")
        if getattr(res, "error", None):
            raise RuntimeError(f"Kaggle rejected the upload: {res.error}")
    _wait_dataset_ready(api, ref)
    with _lock:
        item = _find(item_id)
        if item is not None:
            item.update(status="cached", error=None, size_bytes=src.stat().st_size, filename=src.name)
            _save()
    _log(f"Library: '{label}' is cached in {ref}.")


def _delete_dataset(api, ref: str):
    owner, slug = ref.split("/", 1)
    try:
        api.dataset_delete(owner, slug, no_confirm=True)
    except Exception as e:
        if "404" not in str(e) and "not found" not in str(e).lower():
            raise


def _do_delete(item_id: str):
    owner, api = _owner_and_api()
    with _lock:
        item = _find(item_id)
        if item is None:
            return
        ref = item.get("dataset") if not item.get("legacy") else None
    if ref:
        _log(f"Library: deleting dataset {ref}...")
        _delete_dataset(api, ref)
    with _lock:
        _data["models"] = [i for i in _data["models"] if i["id"] != item_id]
        _data["loras"] = [i for i in _data["loras"] if i["id"] != item_id]
        if _data.get("active_model") == item_id:
            _data["active_model"] = _data["models"][0]["id"] if _data["models"] else None
        _save()
    if item.get("kind") == "lora" and item.get("filename"):
        (LORAS_DIR / item["filename"]).unlink(missing_ok=True)
    _log(f"Library: removed '{item['name']}'.")


def _do_delete_legacy(ref: str):
    _, api = _owner_and_api()
    _log(f"Library: deleting old dataset {ref}...")
    _delete_dataset(api, ref)
    with _lock:
        _data["legacy"] = [d for d in _data["legacy"] if d["ref"].lower() != ref.lower()]
        for item in _all_items():
            if item.get("legacy") and (item.get("dataset") or "").lower() == ref.lower():
                item.update(dataset=None, legacy=False, status="not_cached")
        _save()
    _log(f"Library: deleted {ref}.")


# ---------------------------------------------------------------- API

class AddModelRequest(BaseModel):
    url: str
    name: str = ""


class AddLoraRequest(BaseModel):
    url: str
    name: str = ""
    scale: float = 1.0
    is_turbo: bool = False
    description: str = ""
    civitai_token: str = ""


class UpdateLoraRequest(BaseModel):
    id: str
    scale: Optional[float] = None
    enabled: Optional[bool] = None


class ActiveModelRequest(BaseModel):
    id: str


def _bad_request(e: Exception):
    raise HTTPException(status_code=400, detail=str(e))


@router.get("/library")
def get_library():
    return snapshot()


@router.post("/library/refresh")
async def refresh_library():
    import asyncio
    try:
        return await asyncio.to_thread(refresh)
    except Exception as e:
        _bad_request(e)


@router.post("/library/models/add")
async def add_model(req: AddModelRequest):
    import asyncio
    try:
        repo, revision = parse_model_source(req.url)
        info = await asyncio.to_thread(validate_model_repo, repo, revision)
    except Exception as e:
        _bad_request(e)
    with _lock:
        if any(m["source"].lower() == repo.lower() and m.get("revision") == revision for m in _data["models"]):
            raise HTTPException(status_code=400, detail=f"'{repo}' is already in the library.")
        name = req.name.strip() or repo.split("/")[-1].replace("-", " ")
        item = _item(kind="model", id=_unique_id(name), name=name, source=repo, revision=revision,
                     size_bytes=info["size_bytes"], description=f"From huggingface.co/{repo}")
        _data["models"].append(item)
        _save()
    _enqueue(("cache", item["id"]))
    return {"status": "queued", "item": item}


@router.post("/library/models/active")
def set_active_model(req: ActiveModelRequest):
    with _lock:
        item = _find(req.id)
        if item is None or item["kind"] != "model":
            raise HTTPException(status_code=404, detail="Model not found")
        _data["active_model"] = req.id
        _save()
    return snapshot()


@router.post("/library/items/{item_id}/cache")
def cache_item(item_id: str):
    with _lock:
        item = _find(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        if item["status"] == "caching":
            raise HTTPException(status_code=400, detail="Already caching.")
        if item["kind"] == "lora" and not item["source"].startswith("http") and item["source"].count("/") < 2:
            local = LORAS_DIR / (item.get("filename") or "")
            if not local.is_file():
                raise HTTPException(status_code=400, detail="This LoRA was uploaded from a PC and the file isn't here any more. Upload it again.")
            item["status"] = "caching"
            _save()
            _enqueue(("upload", item_id, str(local)))
            return {"status": "queued"}
        item["status"] = "caching"
        _save()
        download_url = parse_lora_source(item["source"])["download_url"] if item["kind"] == "lora" else ""
    _enqueue(("cache", item_id, download_url))
    return {"status": "queued"}


@router.delete("/library/items/{item_id}")
def delete_item(item_id: str):
    with _lock:
        item = _find(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        if item["status"] == "caching":
            raise HTTPException(status_code=400, detail="Wait until caching finishes before deleting.")
        if item["kind"] == "model" and len(_data["models"]) == 1:
            raise HTTPException(status_code=400, detail="Keep at least one model in the library.")
    _enqueue(("delete", item_id))
    return {"status": "queued"}


@router.delete("/library/legacy/{owner}/{slug}")
def delete_legacy(owner: str, slug: str):
    ref = f"{owner}/{slug}"
    with _lock:
        if not any(d["ref"].lower() == ref.lower() for d in _data.get("legacy", [])):
            raise HTTPException(status_code=404, detail="Not one of the app's old datasets")
    _enqueue(("delete_legacy", ref))
    return {"status": "queued"}


# LoRA routes keep their old paths so the LoRA stack UI works unchanged.

@router.get("/loras")
def get_loras():
    return {"loras": snapshot()["loras"]}


@router.post("/loras/update")
def update_lora(req: UpdateLoraRequest):
    with _lock:
        item = _find(req.id)
        if item is None or item["kind"] != "lora":
            raise HTTPException(status_code=404, detail="LoRA not found")
        if req.scale is not None:
            item["scale"] = float(req.scale)
        if req.enabled is not None:
            item["enabled"] = bool(req.enabled)
        _save()
        return {"status": "success", "lora": dict(item, usable=is_usable(item))}


@router.post("/loras/add")
async def add_lora(req: AddLoraRequest):
    import asyncio
    try:
        info = parse_lora_source(req.url)
    except Exception as e:
        _bad_request(e)
    with _lock:
        if any(l["source"] == info["source"] for l in _data["loras"]):
            raise HTTPException(status_code=400, detail="This LoRA is already in the library.")

    if req.civitai_token and "civitai" in req.url:
        # With a personal Civitai key the download runs here, so the key never leaves this PC.
        dl, fname, cname = await asyncio.to_thread(resolve_civitai, req.url, req.civitai_token)
        local = await asyncio.to_thread(_download_locally, dl, fname, req.civitai_token)
        with _lock:
            name = req.name.strip() or cname or local.stem.replace("_", " ")
            item = _item(kind="lora", id=_unique_id(local.stem), name=name, source=req.url, filename=local.name,
                         scale=req.scale, enabled=True, is_turbo=req.is_turbo,
                         description=req.description or f"From {urlparse(req.url).netloc}")
            item["status"] = "caching"
            _data["loras"].append(item)
            _save()
        _enqueue(("upload", item["id"], str(local)))
        return {"status": "queued", "lora": item}

    with _lock:
        name = req.name.strip() or (info["name"] or "custom lora").replace("_", " ").replace("-", " ")
        item = _item(kind="lora", id=_unique_id(Path(info["filename"] or name).stem), name=name,
                     source=info["source"], filename=info["filename"] or None, scale=req.scale, enabled=True,
                     is_turbo=req.is_turbo, description=req.description or f"From {urlparse(info['source']).netloc or 'huggingface.co'}")
        item["status"] = "caching"
        _data["loras"].append(item)
        _save()
    _enqueue(("cache", item["id"], info["download_url"]))
    return {"status": "queued", "lora": item}


def _download_locally(url: str, filename: str, civitai_token: str) -> Path:
    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {civitai_token}"}
    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        if r.status_code in (401, 403) or "/login" in r.url:
            raise HTTPException(status_code=400, detail="Civitai refused the download. Check your Civitai API key.")
        r.raise_for_status()
        if not filename:
            m = re.search(r"filename\*?=(?:UTF-8'')?[\"']?([^\"';\r\n]+)", r.headers.get("content-disposition", ""), re.I)
            filename = unquote(m.group(1)).strip() if m else "civitai_lora.safetensors"
        dest = LORAS_DIR / Path(filename).name
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if dest.suffix != ".safetensors":
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The download is not a .safetensors file.")
    return dest


@router.post("/loras/upload")
async def upload_lora(
    file: UploadFile = File(...),
    name: Optional[str] = Form(""),
    scale: float = Form(1.0),
    is_turbo: bool = Form(False),
    description: Optional[str] = Form(""),
):
    filename = Path((file.filename or "").replace("\\", "/")).name
    if not filename.endswith(".safetensors") or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Only .safetensors files are supported for LoRAs.")
    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    dest = LORAS_DIR / filename
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with _lock:
        item = next((l for l in _data["loras"] if l.get("filename") == filename), None)
        if item is None:
            item = _item(kind="lora", id=_unique_id(Path(filename).stem), source=filename, filename=filename,
                         name=name or Path(filename).stem.replace("_", " "),
                         description=description or f"Uploaded from PC: {filename}")
            _data["loras"].append(item)
        item.update(scale=scale, enabled=True, is_turbo=is_turbo, status="caching", error=None)
        if name:
            item["name"] = name
        _save()
    _enqueue(("upload", item["id"], str(dest)))
    return {"status": "queued", "lora": item}


@router.delete("/loras/{lora_id}")
def delete_lora(lora_id: str):
    return delete_item(lora_id)
