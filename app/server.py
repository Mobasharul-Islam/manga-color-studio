import io
import os
import re
import sys
import glob
import json
import time
import shutil
import asyncio
import threading
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from PIL import Image

from scripts.common import (
    ACCELERATORS, IMAGE_EXTS, SCANS_DATASET_SLUG, natural_key, safe_name, resolve_kaggle_username,
)
from app import library, postprocess

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
INPUT_DIR = BASE_DIR / "input_pages"
OUTPUT_DIR = BASE_DIR / "output_app"
SCRIPTS_DIR = BASE_DIR / "scripts"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Manga Colorization TPU Studio")

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/image/thumb/"):
            return response  # versioned URLs (?v=mtime): safe to cache
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)
# No CORS middleware on purpose: the UI is served from this same origin, and allowing
# other origins would let any website read tokens or files through this local server.

# Global Job State
job_state = {
    "status": "idle",       # idle, preparing, pushing, queued, running, downloading, completed, error
    "phase_message": "Ready to colorize",
    "chapter_name": "chapter_01",
    "total_images": 0,
    "completed_images": 0,
    "start_time": 0,
    "elapsed_seconds": 0,
    "logs": [],
    "error": None,
    "input_images": [],
    "output_images": [],
    "last_kernel_status": None,
}

state_lock = threading.Lock()


def log_message(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with state_lock:
        job_state["logs"].append(formatted)
        if len(job_state["logs"]) > 200:
            job_state["logs"].pop(0)


# Request Models
class LoraConfig(BaseModel):
    id: Optional[str] = None
    name: str
    path: str
    filename: Optional[str] = None
    scale: float = 1.0
    enabled: bool = True
    is_turbo: bool = False
    is_custom: bool = False
    description: str = ""


class JobStartRequest(BaseModel):
    chapter_name: str = "chapter_01"
    max_dim: int = 1440
    num_steps: int = 8
    guidance_scale: float = 1.0
    prompt: str = "masterpiece, best quality, anime style, highly detailed coloring, vibrant color harmony, crisp lineart"
    negative_prompt: str = "worst quality, low quality, blurry, artifact, oversaturated, deformed, greyscale, monochrome"
    loras: List[LoraConfig] = []  # ignored: LoRAs and model come from the library
    # Original scan file names to re-run within an existing chapter; empty = whole chapter.
    pages: List[str] = []
    # Key of scripts.common.ACCELERATORS: "tpu", "gpu_t4" or "gpu_p100".
    accelerator: str = "tpu"


def _image_item(path: Path) -> Dict[str, Any]:
    return {
        "name": path.name,
        "path": str(path),
        "size": path.stat().st_size,
        "url": f"/api/image/input/{path.name}",
        "thumb_url": f"/api/image/thumb/loaded/input/{path.name}?v={int(path.stat().st_mtime)}",
    }


def _reset_input_dir():
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)


def _unique_input_name(base_name: str, parent: str = "") -> str:
    """Keeps both files when two subfolders contain the same name (e.g. 001.jpg)."""
    if not (INPUT_DIR / base_name).exists():
        return base_name
    prefix = safe_name(parent, "page")
    candidate = f"{prefix}_{base_name}"
    i = 2
    while (INPUT_DIR / candidate).exists():
        candidate = f"{prefix}_{i}_{base_name}"
        i += 1
    return candidate


def open_file_dialog_thread():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder_selected = filedialog.askdirectory(title="Select Manga Chapter Folder")
        root.destroy()
        return folder_selected
    except Exception as e:
        print("Tkinter dialog error:", e)
        return ""


@app.post("/api/select-folder")
async def select_folder():
    """Opens native Windows folder picker and loads images."""
    folder = await asyncio.to_thread(open_file_dialog_thread)
    if not folder:
        return {"status": "cancelled", "images": []}

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    found_files = []
    for f in sorted(os.listdir(folder)):
        if Path(f).suffix.lower() in image_extensions:
            found_files.append(os.path.join(folder, f))

    if not found_files:
        raise HTTPException(status_code=400, detail=f"No image files found in '{folder}'")

    # Clear current input directory and copy files
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied_images = []
    for src in found_files:
        dst = INPUT_DIR / Path(src).name
        shutil.copy2(src, dst)
        copied_images.append(_image_item(dst))

    with state_lock:
        job_state["chapter_name"] = Path(folder).name or "chapter_01"
        job_state["input_images"] = copied_images
        job_state["total_images"] = len(copied_images)

    log_message(f"Loaded {len(copied_images)} pages from '{folder}'")
    return {
        "status": "success",
        "folder": folder,
        "chapter_name": Path(folder).name,
        "images": copied_images
    }


@app.post("/api/upload-zip")
async def upload_zip_endpoint(file: UploadFile = File(...)):
    """Extracts an uploaded .zip or .cbz manga chapter archive directly into INPUT_DIR."""
    fname_lower = (file.filename or "").lower()
    if not (fname_lower.endswith(".zip") or fname_lower.endswith(".cbz")):
        raise HTTPException(status_code=400, detail="Please upload a valid .zip or .cbz manga chapter archive.")

    _reset_input_dir()

    temp_zip = BASE_DIR / f"temp_upload_{int(time.time())}.zip"
    try:
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        import zipfile
        uploaded = []
        raw_stem = Path(file.filename).stem
        detected_chapter = raw_stem

        with zipfile.ZipFile(temp_zip, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename.replace("\\", "/"))
                if member_path.suffix.lower() in IMAGE_EXTS:
                    base_name = member_path.name
                    # Ignore hidden / system files
                    if base_name.startswith("._") or base_name.startswith("."):
                        continue
                    dst = INPUT_DIR / _unique_input_name(base_name, member_path.parent.name)
                    with zf.open(member) as source, open(dst, "wb") as target:
                        shutil.copyfileobj(source, target)
                    uploaded.append(_image_item(dst))

        if not uploaded:
            raise HTTPException(status_code=400, detail="No valid image files found inside the archive.")

        uploaded = sorted(uploaded, key=lambda item: natural_key(item["name"]))

        clean_ch = safe_name(detected_chapter)
        with state_lock:
            job_state["chapter_name"] = clean_ch
            job_state["input_images"] = uploaded
            job_state["total_images"] = len(uploaded)

        log_message(f"Extracted and loaded {len(uploaded)} manga pages from '{file.filename}'.")
        return {
            "status": "success",
            "chapter_name": clean_ch,
            "images": uploaded
        }
    finally:
        if temp_zip.exists():
            try:
                temp_zip.unlink()
            except Exception:
                pass


@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Handles drag-and-drop or folder-picker uploaded image files or zip archives."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # If a single .zip or .cbz file was uploaded through upload-images
    if len(files) == 1 and (files[0].filename.lower().endswith(".zip") or files[0].filename.lower().endswith(".cbz")):
        return await upload_zip_endpoint(files[0])

    _reset_input_dir()
    uploaded = []
    detected_chapter = None

    for file in files:
        if not file.filename:
            continue
        # Extract clean basename and handle nested paths from webkitdirectory
        raw_filename = file.filename.replace("\\", "/")
        path_parts = [p for p in raw_filename.split("/") if p]
        if len(path_parts) > 1 and not detected_chapter:
            detected_chapter = path_parts[0]
            
        base_name = path_parts[-1]
        if Path(base_name).suffix.lower() not in IMAGE_EXTS:
            continue

        parent = path_parts[-2] if len(path_parts) > 1 else ""
        dst = INPUT_DIR / _unique_input_name(base_name, parent)
        with open(dst, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded.append(_image_item(dst))

    uploaded = sorted(uploaded, key=lambda item: natural_key(item["name"]))

    with state_lock:
        if detected_chapter:
            job_state["chapter_name"] = safe_name(detected_chapter)
        job_state["input_images"] = uploaded
        job_state["total_images"] = len(uploaded)

    log_message(f"Uploaded {len(uploaded)} manga pages.")
    return {
        "status": "success",
        "chapter_name": job_state.get("chapter_name", "chapter_01"),
        "images": uploaded
    }


@app.get("/api/files")
def get_current_files():
    images = []
    for f in _sorted_images(INPUT_DIR):
        images.append(_image_item(f))
    return {"images": images, "chapter_name": job_state["chapter_name"]}


@app.post("/api/clear-files")
def clear_files():
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    with state_lock:
        job_state["input_images"] = []
        job_state["total_images"] = 0
    return {"status": "cleared"}


class SaveSettingsRequest(BaseModel):
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    hf_token: Optional[str] = None


_kaggle_api = None


def get_kaggle_api_safe(verify_online: bool = False):
    """Authenticated KaggleApi, or None. The client is cached (authenticating takes ~1.7 s);
    verify_online=True rebuilds it from the current credentials and checks them online."""
    global _kaggle_api
    if _kaggle_api is not None and not verify_online:
        return _kaggle_api
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        if verify_online:
            u = getattr(api, "config_values", {}).get("username") or os.getenv("KAGGLE_USERNAME")
            api.dataset_list(user=u)
        _kaggle_api = api
        return api
    except Exception:
        _kaggle_api = None
        return None


def get_kaggle_username() -> str:
    """Dynamically resolves the authenticated Kaggle username (raises if not configured)."""
    return resolve_kaggle_username(get_kaggle_api_safe())


def require_kaggle_username() -> str:
    try:
        return get_kaggle_username()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/settings")
def get_settings_endpoint():
    """Returns the current Kaggle and Hugging Face authentication status."""
    api = get_kaggle_api_safe()
    is_authenticated = api is not None
    username = None
    if is_authenticated:
        try:
            username = get_kaggle_username()
        except RuntimeError:
            is_authenticated = False

    hf_tok = os.getenv("HF_TOKEN", "")
    return {
        "kaggle_authenticated": is_authenticated,
        "kaggle_username": username,
        "hf_token_configured": bool(hf_tok),
        "hf_token_preview": (hf_tok[:6] + "..." + hf_tok[-4:]) if (hf_tok and len(hf_tok) > 10) else ""
    }


@app.post("/api/settings/save")
def save_settings_endpoint(req: SaveSettingsRequest):
    """Saves Kaggle credentials and HF token, verifying connection immediately."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_file = kaggle_dir / "kaggle.json"

    key_or_token = req.kaggle_key.strip() if req.kaggle_key else ""
    username = req.kaggle_username.strip() if req.kaggle_username else ""

    if key_or_token and not username:
        try:
            username = resolve_kaggle_username()
        except RuntimeError:
            raise HTTPException(status_code=400, detail="Please enter your Kaggle username.")

    if key_or_token:
        # Clear legacy cached access tokens
        (kaggle_dir / "access_token").unlink(missing_ok=True)
        (kaggle_dir / "access_token.txt").unlink(missing_ok=True)
        (kaggle_dir / "token").unlink(missing_ok=True)

        if key_or_token.startswith("KGAT_"):
            (kaggle_dir / "access_token").write_text(key_or_token, encoding="utf-8")
            os.environ["KAGGLE_API_TOKEN"] = key_or_token
        
        creds = {
            "username": username,
            "key": key_or_token
        }
        kaggle_file.write_text(json.dumps(creds, indent=2), encoding="utf-8")
        if username:
            os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key_or_token

    if req.hf_token:
        os.environ["HF_TOKEN"] = req.hf_token.strip()

    # Update .env file
    env_file = BASE_DIR / ".env"
    env_lines = []
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("KAGGLE_USERNAME=") and username:
                continue
            if line.startswith("KAGGLE_KEY=") and key_or_token:
                continue
            if line.startswith("KAGGLE_API_TOKEN=") and key_or_token:
                continue
            if line.startswith("HF_TOKEN=") and req.hf_token:
                continue
            env_lines.append(line)
    
    if username:
        env_lines.append(f"KAGGLE_USERNAME={username}")
    if key_or_token:
        env_lines.append(f"KAGGLE_KEY={key_or_token}")
        if key_or_token.startswith("KGAT_"):
            env_lines.append(f"KAGGLE_API_TOKEN={key_or_token}")
    if req.hf_token:
        env_lines.append(f"HF_TOKEN={req.hf_token.strip()}")

    env_file.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    # Verify Kaggle authentication
    api = get_kaggle_api_safe(verify_online=True)
    if api:
        u = get_kaggle_username()
        log_message(f"Kaggle API authenticated successfully as @{u}")
        return {
            "status": "success",
            "kaggle_authenticated": True,
            "kaggle_username": u,
            "message": f"Successfully connected as @{u}!"
        }
    else:
        log_message("Kaggle API authentication failed with provided credentials.")
        return {
            "status": "warning",
            "kaggle_authenticated": False,
            "error": "Authentication failed",
            "message": "Saved credentials, but Kaggle online authentication failed. Please verify your username and API key/token."
        }


@app.post("/api/settings/upload-kaggle-json")
async def upload_kaggle_json_endpoint(file: UploadFile = File(...)):
    """Uploads and configures a kaggle.json file directly from the browser."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Please upload a valid kaggle.json file.")

    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
        if "username" not in data or "key" not in data:
            raise HTTPException(status_code=400, detail="Invalid kaggle.json format: missing username or key.")
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        (kaggle_dir / "access_token").unlink(missing_ok=True)
        (kaggle_dir / "access_token.txt").unlink(missing_ok=True)
        (kaggle_dir / "token").unlink(missing_ok=True)

        kaggle_file = kaggle_dir / "kaggle.json"
        kaggle_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        os.environ["KAGGLE_USERNAME"] = data["username"]
        os.environ["KAGGLE_KEY"] = data["key"]

        # Update .env
        env_file = BASE_DIR / ".env"
        env_lines = []
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("KAGGLE_USERNAME=") or line.startswith("KAGGLE_KEY="):
                    continue
                env_lines.append(line)
        env_lines.append(f"KAGGLE_USERNAME={data['username']}")
        env_lines.append(f"KAGGLE_KEY={data['key']}")
        env_file.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

        # Verify
        api = get_kaggle_api_safe(verify_online=True)
        if not api:
            raise RuntimeError("Online verification failed with the uploaded kaggle.json")
        u = data["username"]
        log_message(f"Uploaded kaggle.json authenticated successfully as @{u}")
        return {
            "status": "success",
            "kaggle_authenticated": True,
            "kaggle_username": u,
            "message": f"Successfully connected as @{u}!"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process kaggle.json: {e}")


DATASET_READY_TIMEOUT_S = 600          # Kaggle indexing of a freshly uploaded scans dataset
KERNEL_TIMEOUT_S = 10 * 3600           # Kaggle TPU sessions are capped at 9 hours
MAX_STATUS_FAILURES = 8                # consecutive failed `kernels status` calls (~2 minutes)

# Per-chapter layout under output_app/<chapter>/:
#   input/       original scans, snapshotted when the chapter is colorized
#   generated/   raw TPU output (edit_<stem>.png)
#   final/       post-processed pages (lineart blend / bubble restore)
#   history/     raw outputs replaced by a later re-run, one folder per run
#   postprocess.json


def chapter_dir_for(chapter: str) -> Path:
    return OUTPUT_DIR / safe_name(chapter)


def _sorted_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def pair_originals(chapter_dir: Path) -> Dict[str, Tuple[Path, bool]]:
    """{generated file name: (original scan, matched_by_name)} from the chapter's input snapshot.

    Pages the app colorizes are named edit_<scan stem>.png and match by name. Chapters put
    together outside the app (e.g. ComfyUI's 0_00001_.png) can't match by name; when no page
    does and both folders hold the same number of images, they are paired in sorted order.
    """
    inputs = _sorted_images(chapter_dir / "input")
    colors = sorted((chapter_dir / "generated").glob("*.png"), key=lambda f: natural_key(f.name))
    by_stem = {f.stem: f for f in inputs}
    pairs = {}
    for c in colors:
        stem = c.stem[len("edit_"):] if c.stem.startswith("edit_") else c.stem
        if stem in by_stem:
            pairs[c.name] = (by_stem[stem], True)
    if not pairs and inputs and len(inputs) == len(colors):
        pairs = {c.name: (i, False) for c, i in zip(colors, inputs)}
    return pairs


def find_original(chapter_dir: Path, color_name: str) -> Optional[Path]:
    """Original scan for one generated page, or None."""
    match = pair_originals(chapter_dir).get(Path(color_name).name)
    return match[0] if match else None


def load_post_settings(chapter_dir: Path) -> postprocess.PostSettings:
    try:
        data = json.loads((chapter_dir / "postprocess.json").read_text(encoding="utf-8"))
        return postprocess.PostSettings(
            lineart_strength=float(data.get("lineart_strength", 0.5)),
            preserve_bubbles=bool(data.get("preserve_bubbles", False)),
        )
    except Exception:
        return postprocess.PostSettings()


def postprocess_chapter(chapter_dir: Path, settings: postprocess.PostSettings, only: Optional[List[str]] = None) -> int:
    """Writes final/<page> for every generated page that has an original scan. Returns the count."""
    final_dir = chapter_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    pairs = pair_originals(chapter_dir)
    for color_file in sorted((chapter_dir / "generated").glob("*.png")):
        if only is not None and color_file.name not in only:
            continue
        dst = final_dir / color_file.name
        original = pairs.get(color_file.name, (None, False))[0]
        if settings.is_noop or original is None:
            dst.unlink(missing_ok=True)
            continue
        with Image.open(color_file) as color, Image.open(original) as orig:
            postprocess.apply(color, orig, settings).save(dst)
        count += 1
    return count


def merge_outputs(chapter_dir: Path, new_files: List[Path]) -> int:
    """Moves freshly downloaded pages into generated/, backing up any page they replace."""
    generated = chapter_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    history = chapter_dir / "history" / time.strftime("%Y%m%d-%H%M%S")
    replaced = 0
    for f in new_files:
        dst = generated / f.name
        if dst.exists():
            history.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(history / f.name))
            replaced += 1
        shutil.move(str(f), str(dst))
    return replaced


def upload_scans_dataset(api, kaggle_user: str, scans: List[Path], chapter_name: str):
    """Zips the scans into the private scans dataset and waits until Kaggle has indexed it."""
    import zipfile

    dataset_full_slug = f"{kaggle_user}/{SCANS_DATASET_SLUG}"
    with state_lock:
        job_state["phase_message"] = "Packaging & uploading scans to Kaggle..."
    log_message(f"Packaging {len(scans)} scans for Kaggle dataset '{dataset_full_slug}'...")

    staging_dir = BASE_DIR / "kaggle_datasets" / "input_pages_staging"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(staging_dir / "scans.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for scan in scans:
            zf.write(scan, arcname=scan.name)
    with open(staging_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump({"title": SCANS_DATASET_SLUG, "id": dataset_full_slug, "licenses": [{"name": "CC0-1.0"}]}, f)

    existing_datasets = []
    try:
        existing_datasets = [getattr(d, "ref", "") or getattr(d, "id", "") for d in api.dataset_list(user=kaggle_user)]
    except Exception as e:
        log_message(f"Note checking existing datasets: {e}")
    dataset_exists = any(dataset_full_slug.lower() == d.lower() for d in existing_datasets)

    if dataset_exists:
        try:
            log_message(f"Updating existing Kaggle dataset '{dataset_full_slug}'...")
            api.dataset_create_version(str(staging_dir), version_notes=f"Chapter: {chapter_name}", dir_mode="zip")
        except Exception as ver_err:
            if "403" in str(ver_err) or "404" in str(ver_err) or "not found" in str(ver_err).lower():
                log_message(f"Version update returned ({ver_err}). Creating new dataset '{dataset_full_slug}'...")
                api.dataset_create_new(str(staging_dir), dir_mode="zip", public=False, quiet=False)
            else:
                raise
    else:
        log_message(f"Creating new private Kaggle dataset '{dataset_full_slug}'...")
        api.dataset_create_new(str(staging_dir), dir_mode="zip", public=False, quiet=False)

    log_message("Kaggle dataset upload submitted. Waiting for Kaggle backend to finish indexing...")
    with state_lock:
        job_state["phase_message"] = "Waiting for Kaggle dataset processing..."

    # Poll until Kaggle marks the version ready, so the kernel never mounts the previous scans.
    poll_interval = 4
    waited = 0
    while waited < DATASET_READY_TIMEOUT_S:
        time.sleep(poll_interval)
        waited += poll_interval
        try:
            ds_status = str(api.dataset_status(dataset_full_slug)).lower()
        except Exception:
            continue  # transient status API errors right after upload
        if "ready" in ds_status:
            log_message(f"Kaggle dataset '{dataset_full_slug}' is ready! ({waited}s)")
            return
        if "error" in ds_status or "fail" in ds_status:
            raise RuntimeError(f"Kaggle dataset processing failed: {ds_status}")
        if waited % 12 == 0:
            log_message(f"Dataset indexing on Kaggle... ({waited}s, status: {ds_status})")
            with state_lock:
                job_state["phase_message"] = f"Kaggle indexing dataset ({waited}s)..."
    raise RuntimeError(
        f"Kaggle did not finish indexing '{dataset_full_slug}' within {DATASET_READY_TIMEOUT_S // 60} minutes. "
        "Starting now would colorize the previous scans; please try again in a few minutes."
    )


def monitor_kernel(sys_py: str, kernel_name: str, label: str):
    deadline = time.time() + KERNEL_TIMEOUT_S
    is_running = False
    failures = 0
    while True:
        time.sleep(15)
        if time.time() > deadline:
            raise RuntimeError(f"Kaggle kernel did not finish within {KERNEL_TIMEOUT_S // 3600} hours.")
        status_res = subprocess.run([sys_py, "-m", "kaggle", "kernels", "status", kernel_name], capture_output=True, text=True, cwd=BASE_DIR)
        status_text = (status_res.stdout or "").strip()

        with state_lock:
            job_state["elapsed_seconds"] = int(time.time() - job_state["start_time"])
            job_state["last_kernel_status"] = status_text

        if status_res.returncode != 0 or not status_text:
            failures += 1
            log_message(f"Kernel status check failed ({failures}/{MAX_STATUS_FAILURES}): {(status_res.stderr or '').strip()[:200]}")
            if failures >= MAX_STATUS_FAILURES:
                raise RuntimeError("Lost contact with Kaggle while monitoring the kernel. Check the kernel on kaggle.com.")
            continue
        failures = 0

        if "QUEUED" in status_text:
            with state_lock:
                job_state["status"] = "queued"
                job_state["phase_message"] = f"Queued for a Kaggle {label}..."
            log_message(f"Status: QUEUED ({job_state['elapsed_seconds']}s elapsed)")
        elif "RUNNING" in status_text:
            if not is_running:
                is_running = True
                log_message(f"{label} allocated! Loading the model & colorizing...")
            with state_lock:
                job_state["status"] = "running"
                job_state["phase_message"] = f"Colorizing on {label}..."
            log_message(f"Status: RUNNING ({job_state['elapsed_seconds']}s elapsed)")
        elif "COMPLETE" in status_text:
            log_message("Kaggle Kernel Completed Successfully!")
            return
        elif "ERROR" in status_text or "CANCEL" in status_text:
            raise RuntimeError(f"Kaggle kernel failed with status: {status_text}")


def background_tpu_runner(req: JobStartRequest):
    """Executes the full pipeline: Scans Upload -> Notebook Build -> Kaggle Push -> Monitor -> Download."""
    sys_py = sys.executable
    safe_chapter = safe_name(req.chapter_name)
    chapter_dir = OUTPUT_DIR / safe_chapter
    acc = ACCELERATORS[req.accelerator]
    try:
        with state_lock:
            job_state["status"] = "preparing"
            job_state["phase_message"] = "Preparing the notebook..."
            job_state["chapter_name"] = safe_chapter
            job_state["start_time"] = time.time()
            job_state["error"] = None
            job_state["completed_images"] = 0
            job_state["logs"] = []

        kaggle_user = get_kaggle_username()
        api = get_kaggle_api_safe()
        if not api:
            raise RuntimeError("Kaggle API is not authenticated. Please check your Kaggle API key in Settings.")

        log_message("=== Starting Manga Colorization Job ===")
        log_message(f"User: @{kaggle_user} | Chapter: {safe_chapter} | Hardware: {acc['label']} | Max Dim: {req.max_dim}px | Steps: {req.num_steps}")

        # 1. Pick the scans: selected pages of an existing chapter, or the whole loaded chapter
        if req.pages:
            source_dir = chapter_dir / "input"
            wanted = sorted({Path(p).name for p in req.pages}, key=natural_key)
            scans = [source_dir / n for n in wanted if (source_dir / n).is_file()]
            missing = [n for n in wanted if not (source_dir / n).is_file()]
            if missing:
                raise RuntimeError(f"Original scans not found for: {', '.join(missing)}")
            log_message(f"Re-running {len(scans)} selected page(s): {', '.join(wanted)}")
        else:
            scans = _sorted_images(INPUT_DIR)
            if not scans:
                raise RuntimeError("No scans loaded. Select a folder or drop a .zip/.cbz first.")
            snapshot = chapter_dir / "input"
            shutil.rmtree(snapshot, ignore_errors=True)
            snapshot.mkdir(parents=True, exist_ok=True)
            for f in scans:
                shutil.copy2(f, snapshot / f.name)

        with state_lock:
            job_state["total_images"] = len(scans)

        upload_scans_dataset(api, kaggle_user, scans, safe_chapter)

        # 2. Build lightweight notebook with the library's model, enabled LoRAs & settings
        from scripts.make_notebook import build_notebook
        spec = library.run_spec()
        log_message(f"Model: {spec['model']['name']} ({'cached dataset' if spec['model']['dataset'] else 'download from Hugging Face'})")
        for l in spec["loras"]:
            log_message(f"LoRA: {l['name']} @ {l['scale']:.2f} ({'cached dataset' if l['dataset'] else 'download'})")
        build_notebook(
            input_dir=str(INPUT_DIR),
            model=spec["model"],
            loras=spec["loras"],
            max_dim=req.max_dim,
            num_steps=req.num_steps,
            guidance_scale=req.guidance_scale,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            accelerator=req.accelerator,
        )
        log_message("Notebook successfully compiled.")

        # 3. Push to Kaggle
        with state_lock:
            job_state["status"] = "pushing"
            job_state["phase_message"] = f"Pushing the notebook to Kaggle ({acc['label']})..."
        log_message(f"Pushing notebook {kaggle_user}/{acc['slug']} ({acc['label']})...")

        push_res = subprocess.run([sys_py, "-m", "kaggle", "kernels", "push", "-p", acc["dir"]], capture_output=True, text=True, cwd=BASE_DIR)
        log_message(push_res.stdout.strip())
        # The CLI can exit 0 while printing "Kernel push error: ..."
        if push_res.returncode != 0 or "error" in (push_res.stdout or "").lower():
            raise RuntimeError(f"Kaggle push failed: {(push_res.stderr or push_res.stdout).strip()}")

        # 4. Monitor Kaggle Kernel execution
        monitor_kernel(sys_py, f"{kaggle_user}/{acc['slug']}", acc["label"])

        # 5. Download results into a scratch folder, then merge into the chapter
        with state_lock:
            job_state["status"] = "downloading"
            job_state["phase_message"] = "Downloading finished high-res color images..."
        download_dir = chapter_dir / "_download"
        shutil.rmtree(download_dir, ignore_errors=True)
        log_message(f"Downloading outputs to {chapter_dir}...")

        dl_res = subprocess.run([sys_py, str(SCRIPTS_DIR / "download_fixed.py"), str(download_dir), kaggle_user, acc["slug"]], capture_output=True, text=True, cwd=BASE_DIR)
        if dl_res.stdout:
            log_message(dl_res.stdout[-300:])
        if dl_res.returncode != 0:
            raise RuntimeError(f"Downloading results failed: {(dl_res.stderr or '').strip()[-500:]}")

        for log_file in download_dir.glob("*.log"):
            shutil.copy2(log_file, chapter_dir / log_file.name)
        new_files = sorted((download_dir / "generated").glob("*.png"))
        if not new_files:
            raise RuntimeError(f"The kernel finished but produced no images. See the logs in {chapter_dir}.")
        replaced = merge_outputs(chapter_dir, new_files)
        shutil.rmtree(download_dir, ignore_errors=True)
        if replaced:
            log_message(f"Previous versions of {replaced} page(s) saved under {chapter_dir / 'history'}.")
        if len(new_files) < len(scans):
            log_message(f"WARNING: only {len(new_files)} of {len(scans)} pages came back. Re-run the missing pages from the gallery.")

        # 6. Post-process the new pages with the chapter's settings
        new_names = [f.name for f in new_files]
        with state_lock:
            job_state["phase_message"] = "Restoring lineart & speech bubbles..."
        postprocess_chapter(chapter_dir, load_post_settings(chapter_dir), only=new_names)

        with state_lock:
            job_state["status"] = "completed"
            job_state["phase_message"] = "Colorization Complete!"
            job_state["completed_images"] = len(new_files)
            job_state["output_images"] = new_names

        log_message(f"SUCCESS: Generated {len(new_files)} colorized pages in {job_state['elapsed_seconds']}s total!")

    except Exception as e:
        err_msg = str(e)
        log_message(f"ERROR: {err_msg}")
        with state_lock:
            job_state["status"] = "error"
            job_state["phase_message"] = "Colorization Failed"
            job_state["error"] = err_msg


@app.post("/api/start")
def start_colorization_job(req: JobStartRequest, background_tasks: BackgroundTasks):
    if req.accelerator not in ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"Unknown hardware '{req.accelerator}'.")
    try:
        library.run_spec()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    with state_lock:
        if job_state["status"] in ["preparing", "pushing", "queued", "running", "downloading"]:
            raise HTTPException(status_code=400, detail="A colorization job is already in progress.")
        # Claim the slot now so a double click cannot start two jobs.
        job_state["status"] = "preparing"
        job_state["start_time"] = time.time()

    background_tasks.add_task(background_tpu_runner, req)
    return {"status": "started", "chapter": req.chapter_name}


@app.get("/api/accelerators")
def get_accelerators():
    """Hardware options with the account's remaining weekly Kaggle quota."""
    quota = {}
    api = get_kaggle_api_safe()
    if api is not None:
        try:
            res = api.quota_view()
            refresh = res.quota_refresh_time.isoformat() if res.quota_refresh_time else ""
            for name, q in (("GPU", res.gpu_quota), ("TPU", res.tpu_quota)):
                if q is None:
                    continue
                # timedeltas: use total_seconds() (the SDK's JSON dump drops the days part)
                used = q.time_used.total_seconds() / 3600
                total = q.total_time_allowed.total_seconds() / 3600
                quota[name] = {"used_hours": round(used, 2), "remaining_hours": round(max(0.0, total - used), 2),
                               "total_hours": round(total, 2), "refresh_at": refresh}
        except Exception as e:
            log_message(f"Could not read Kaggle quota: {e}")
    return {"accelerators": [
        {"id": key, "label": a["label"], "quota_type": a["quota"], "quota": quota.get(a["quota"])}
        for key, a in ACCELERATORS.items()
    ]}


@app.get("/api/status")
def get_job_status():
    with state_lock:
        if job_state["status"] in ["preparing", "pushing", "queued", "running", "downloading"]:
            job_state["elapsed_seconds"] = int(time.time() - job_state["start_time"])
        return dict(job_state)


def _legacy_input_url(color_file: Path, idx: int, input_files: List[Path]) -> Optional[str]:
    """Best-effort match against input_pages/ for chapters colorized before input snapshots existed."""
    input_map = {}
    for p_path in input_files:
        input_map[p_path.stem] = p_path.name
        num_match = re.search(r'\d+', p_path.stem)
        if num_match:
            num_str = num_match.group(0)
            input_map[num_str.lstrip("0") or "0"] = p_path.name
            input_map[f"{int(num_str):02d}"] = p_path.name

    stem = color_file.stem
    clean_stem = stem.replace("edit_", "").replace("color_", "").replace("out_", "")
    if clean_stem in input_map:
        return f"/api/image/input/{input_map[clean_stem]}"
    if stem in input_map:
        return f"/api/image/input/{input_map[stem]}"
    num_match = re.search(r'\d+', clean_stem)
    if num_match and (num_match.group(0).lstrip("0") or "0") in input_map:
        return f"/api/image/input/{input_map[num_match.group(0).lstrip('0') or '0']}"
    for in_stem, in_name in input_map.items():
        if in_stem == clean_stem or in_stem.startswith(clean_stem) or clean_stem.startswith(in_stem):
            return f"/api/image/input/{in_name}"
    if idx < len(input_files):
        return f"/api/image/input/{input_files[idx].name}"
    return None


@app.get("/api/results")
def get_results(chapter: Optional[str] = None):
    # Discover all available chapters in output_app
    available_chapters = []
    if OUTPUT_DIR.exists():
        for ch_dir in OUTPUT_DIR.iterdir():
            if ch_dir.is_dir() and (ch_dir / "generated").exists() and list((ch_dir / "generated").glob("*.png")):
                available_chapters.append(ch_dir.name)
    available_chapters.sort(key=natural_key)

    # Pick requested chapter, or current job chapter, or first available chapter
    selected_chapter = chapter
    if not selected_chapter:
        curr_job_ch = job_state.get("chapter_name")
        if curr_job_ch and curr_job_ch in available_chapters:
            selected_chapter = curr_job_ch
        elif available_chapters:
            # most recently colorized chapter
            selected_chapter = max(available_chapters, key=lambda c: (OUTPUT_DIR / c / "generated").stat().st_mtime)
        else:
            selected_chapter = "Chapter_1"

    safe_chapter = safe_name(selected_chapter)
    chapter_dir = OUTPUT_DIR / safe_chapter
    generated_dir = chapter_dir / "generated"
    has_snapshot = (chapter_dir / "input").is_dir()
    originals = pair_originals(chapter_dir) if has_snapshot else {}
    legacy_inputs = [] if has_snapshot else _sorted_images(INPUT_DIR)

    pairs = []
    color_files = sorted(generated_dir.glob("*.png"), key=lambda p: natural_key(p.name)) if generated_dir.exists() else []
    for idx, color_file in enumerate(color_files):
        name = color_file.name
        final_file = chapter_dir / "final" / name
        shown = final_file if final_file.exists() else color_file
        kind = "final" if shown is final_file else "generated"
        version = int(shown.stat().st_mtime)

        original, by_name = originals.get(name, (None, False))
        if original is not None:
            input_url = f"/api/image/chapter/{safe_chapter}/input/{original.name}"
        else:
            input_url = _legacy_input_url(color_file, idx, legacy_inputs)

        pairs.append({
            "id": color_file.stem.replace("edit_", ""),
            "color_name": name,
            "color_url": f"/api/image/chapter/{safe_chapter}/{kind}/{name}?v={version}",
            "raw_url": f"/api/image/chapter/{safe_chapter}/generated/{name}?v={int(color_file.stat().st_mtime)}",
            "thumb_url": f"/api/image/thumb/{safe_chapter}/{kind}/{name}?v={version}",
            "input_url": input_url or f"/api/image/chapter/{safe_chapter}/generated/{name}",
            # Lineart/bubble restore needs the original scan.
            "has_original": original is not None,
            # Re-runs write edit_<scan>.png, so they only replace pages matched by name.
            "page": original.name if original is not None and by_name else None,
        })

    settings = load_post_settings(chapter_dir)
    return {
        "chapter_name": safe_chapter,
        "available_chapters": available_chapters,
        "count": len(pairs),
        "pairs": pairs,
        "can_postprocess": bool(originals),
        "postprocess": {"lineart_strength": settings.lineart_strength, "preserve_bubbles": settings.preserve_bubbles},
    }


class PostprocessRequest(BaseModel):
    chapter: str
    lineart_strength: float = 0.5
    preserve_bubbles: bool = False


@app.post("/api/postprocess")
async def postprocess_endpoint(req: PostprocessRequest):
    """Saves a chapter's post-processing settings and re-renders all its final pages."""
    chapter_dir = chapter_dir_for(req.chapter)
    if not (chapter_dir / "generated").is_dir():
        raise HTTPException(status_code=404, detail="Chapter not found")
    if not pair_originals(chapter_dir):
        raise HTTPException(status_code=400, detail="No original scans match this chapter's pages, so it cannot be post-processed.")
    settings = postprocess.PostSettings(
        lineart_strength=max(0.0, min(1.0, req.lineart_strength)),
        preserve_bubbles=req.preserve_bubbles,
    )
    (chapter_dir / "postprocess.json").write_text(
        json.dumps({"lineart_strength": settings.lineart_strength, "preserve_bubbles": settings.preserve_bubbles}, indent=2),
        encoding="utf-8",
    )
    count = await asyncio.to_thread(postprocess_chapter, chapter_dir, settings)
    log_message(f"Post-processed {count} pages of '{chapter_dir.name}' (lineart {settings.lineart_strength:.2f}, bubbles {'on' if settings.preserve_bubbles else 'off'}).")
    return {"status": "success", "count": count}


@app.get("/api/image/preview/{chapter}/{filename}")
def preview_postprocess(chapter: str, filename: str, lineart: float = 0.5, bubbles: bool = False):
    """Renders one page with the given settings without saving, for live slider previews."""
    chapter_dir = chapter_dir_for(chapter)
    name = Path(filename).name
    color_file = chapter_dir / "generated" / name
    original = find_original(chapter_dir, name)
    if not color_file.exists() or original is None:
        raise HTTPException(status_code=404, detail="Page not found")
    settings = postprocess.PostSettings(lineart_strength=max(0.0, min(1.0, lineart)), preserve_bubbles=bubbles)
    with Image.open(color_file) as color, Image.open(original) as orig:
        out = postprocess.apply(color, orig, settings)
    buf = io.BytesIO()
    out.convert("RGB").save(buf, format="JPEG", quality=92)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


THUMB_HEIGHT = 240


@app.get("/api/image/thumb/{chapter}/{kind}/{filename}")
def get_thumbnail(chapter: str, kind: str, filename: str):
    """Small cached JPEG of a page, for page strips and grids (full pages are 1-2 MB each)."""
    name = Path(filename).name
    if chapter == "loaded" and kind == "input":
        src, cache_dir = INPUT_DIR / name, INPUT_DIR / ".thumbs"
    elif kind in {"input", "generated", "final"}:
        src, cache_dir = chapter_dir_for(chapter) / kind / name, chapter_dir_for(chapter) / ".thumbs" / kind
    else:
        raise HTTPException(status_code=404, detail="Unknown image kind")
    if not src.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    thumb = cache_dir / (Path(name).stem + ".jpg")
    if not thumb.exists() or thumb.stat().st_mtime < src.stat().st_mtime:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_HEIGHT * 3, THUMB_HEIGHT))
            im.save(thumb, "JPEG", quality=82)
    return FileResponse(str(thumb), headers={"Cache-Control": "max-age=86400"})


@app.get("/api/image/chapter/{chapter}/{kind}/{filename}")
def get_chapter_image(chapter: str, kind: str, filename: str):
    if kind not in {"input", "generated", "final"}:
        raise HTTPException(status_code=404, detail="Unknown image kind")
    file_path = chapter_dir_for(chapter) / kind / Path(filename).name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(file_path))


@app.get("/api/image/input/{filename:path}")
def get_input_image(filename: str):
    # Only plain file names inside input_pages/; never resolve '..' or subpaths.
    file_path = INPUT_DIR / Path(filename).name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Input image not found")
    return FileResponse(str(file_path))


@app.get("/api/image/output/{chapter}/{filename}")
def get_output_image(chapter: str, filename: str):
    safe_chapter = safe_name(chapter)
    name = Path(filename).name
    file_path = OUTPUT_DIR / safe_chapter / "generated" / name
    if not file_path.exists():
        # Fallback check in out_v31
        alt_path = BASE_DIR / "output_anima" / safe_chapter / "generated" / name
        if alt_path.exists():
            return FileResponse(str(alt_path))
        raise HTTPException(status_code=404, detail="Output image not found")
    return FileResponse(str(file_path))


@app.post("/api/open-folder")
def open_output_folder(chapter_name: Optional[str] = None):
    chapter_dir = chapter_dir_for(chapter_name or job_state.get("chapter_name", "chapter_01"))
    target_dir = OUTPUT_DIR
    for sub in ("final", "generated"):
        candidate = chapter_dir / sub
        if candidate.is_dir() and any(candidate.iterdir()):
            target_dir = candidate
            break
    target_dir.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        os.startfile(str(target_dir))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(target_dir)])
    else:
        subprocess.run(["xdg-open", str(target_dir)])
    return {"status": "opened", "path": str(target_dir)}


library.init(log_message, get_kaggle_api_safe)
app.include_router(library.router)

# Mount static web directory (last, so it never shadows the API routes)
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
