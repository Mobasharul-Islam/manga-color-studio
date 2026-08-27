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
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

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
KERNEL_DIR = BASE_DIR / "kernel_anima"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Manga Colorization TPU Studio")

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# Base directories & paths
LORAS_DIR = BASE_DIR / "loras"
LORAS_DIR.mkdir(parents=True, exist_ok=True)
LORAS_CONFIG = BASE_DIR / "loras.json"

DEFAULT_LORAS = [
    {
        "id": "color_manga",
        "name": "Color Manga Anima (Base Style)",
        "path": "Kutches/Anim4/colorMangaAnima.safetensors",
        "filename": "colorMangaAnima.safetensors",
        "scale": 1.0,
        "enabled": True,
        "is_turbo": False,
        "is_custom": False,
        "description": "Transforms lineart into rich, full-color anime shading."
    },
    {
        "id": "turbo_8step",
        "name": "Anima Turbo (8-Step Fast)",
        "path": "circlestone-labs/Anima-Official-LoRAs/anima-turbo-lora-v0.2.safetensors",
        "filename": "anima-turbo-lora-v0.2.safetensors",
        "scale": 1.0,
        "enabled": True,
        "is_turbo": True,
        "is_custom": False,
        "description": "Accelerates generation down to 8 steps per page on TPU."
    }
]


def load_loras_config() -> List[Dict[str, Any]]:
    staging_dir = BASE_DIR / "kaggle_datasets" / "Loras_Manga_Coloring_Anima"
    staging_dir.mkdir(parents=True, exist_ok=True)

    current_loras = [dict(l) for l in DEFAULT_LORAS]
    if LORAS_CONFIG.exists():
        try:
            with open(LORAS_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
                loaded = data.get("loras", [])
                if loaded:
                    current_loras = loaded
        except Exception as e:
            print("Error reading loras.json:", e)

    # 1. Prune entries if their safetensors file does not exist locally (except built-in defaults)
    existing_files = {p.name for p in LORAS_DIR.glob("*.safetensors")}.union(
        {p.name for p in staging_dir.glob("*.safetensors")}
    )

    filtered_loras = []
    for l in current_loras:
        fname = l.get("filename")
        if fname:
            if fname in existing_files:
                filtered_loras.append(l)
        else:
            filtered_loras.append(l)

    current_loras = filtered_loras

    # 2. Auto-scan local directories & Kaggle dataset folders for any present .safetensors
    known_filenames = {l.get("filename") for l in current_loras if l.get("filename")}
    known_ids = {l.get("id") for l in current_loras if l.get("id")}
    
    scan_dirs = [
        LORAS_DIR,
        staging_dir,
        BASE_DIR / "kaggle_datasets" / "loras_manga_coloring_anima",
        BASE_DIR / "kaggle_datasets" / "loras-manga-coloring-anima",
    ]
    
    for s_dir in scan_dirs:
        if s_dir.exists():
            for sf in s_dir.glob("*.safetensors"):
                # Sync between loras/ and kaggle_datasets/
                dst_local = LORAS_DIR / sf.name
                dst_staging = staging_dir / sf.name
                if not dst_local.exists():
                    shutil.copy2(sf, dst_local)
                if not dst_staging.exists():
                    shutil.copy2(sf, dst_staging)

                if sf.name not in known_filenames:
                    lora_id = re.sub(r'[^a-zA-Z0-9_]', '_', sf.stem).lower()
                    if lora_id in known_ids:
                        lora_id = f"{lora_id}_{int(time.time()) % 1000}"
                    
                    name = sf.stem.replace('_', ' ').replace('-', ' ')
                    new_item = {
                        "id": lora_id,
                        "name": name,
                        "path": sf.name,
                        "filename": sf.name,
                        "scale": 1.0,
                        "enabled": False,
                        "is_turbo": "turbo" in sf.name.lower(),
                        "is_custom": True,
                        "description": f"Dataset LoRA: {sf.name}"
                    }
                    current_loras.append(new_item)
                    known_filenames.add(sf.name)
                    known_ids.add(lora_id)

    save_loras_config(current_loras)
    return current_loras


def save_loras_config(loras_list: List[Dict[str, Any]]) -> None:
    with open(LORAS_CONFIG, "w", encoding="utf-8") as f:
        json.dump({"loras": loras_list}, f, indent=2)


import requests
from urllib.parse import urlparse, unquote, parse_qs
from huggingface_hub import hf_hub_download


def resolve_civitai_download_url(url: str, session: requests.Session, civitai_token: str = "") -> tuple[str, str]:
    """If url is a Civitai model page or API URL, resolves the direct download link and filename."""
    try:
        # Normalize civitai.red, civitai.green, etc. to civitai.com
        url = re.sub(r'https?://civitai\.(red|green|blue|org)/', 'https://civitai.com/', url)
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        model_version_id = qs.get("modelVersionId", [None])[0]

        headers = {}
        if civitai_token:
            headers["Authorization"] = f"Bearer {civitai_token}"

        m = re.search(r'/models/(\d+)', parsed.path)
        if m:
            model_id = m.group(1)
            if model_version_id:
                api_url = f"https://civitai.com/api/v1/model-versions/{model_version_id}"
                r = session.get(api_url, headers=headers, timeout=15)
                if r.status_code == 200:
                    vdata = r.json()
                    dl_url = vdata.get("downloadUrl")
                    files = vdata.get("files", [])
                    fname = files[0].get("name", "") if files else ""
                    name = vdata.get("name", "")
                    return dl_url or f"https://civitai.com/api/download/models/{model_version_id}", fname or name
            else:
                api_url = f"https://civitai.com/api/v1/models/{model_id}"
                r = session.get(api_url, headers=headers, timeout=15)
                if r.status_code == 200:
                    mdata = r.json()
                    versions = mdata.get("modelVersions", [])
                    if versions:
                        v = versions[0]
                        dl_url = v.get("downloadUrl")
                        files = v.get("files", [])
                        fname = files[0].get("name", "") if files else ""
                        name = mdata.get("name", "")
                        return dl_url or f"https://civitai.com/api/download/models/{v.get('id')}", fname or name
    except Exception as e:
        print("Civitai URL resolution note:", e)

    return url, ""


def download_lora_from_url(url_or_path: str, custom_name: str = "", civitai_token: str = "") -> Dict[str, Any]:
    """Downloads a LoRA file from Hugging Face, Civitai, or direct URL into local loras/ folder."""
    url_or_path = url_or_path.strip()
    hf_token = os.getenv("HF_TOKEN") or None
    civitai_token = civitai_token or os.getenv("CIVITAI_TOKEN", "")

    # 1. Hugging Face repo path: e.g. "Kutches/Anim4/colorMangaAnima.safetensors"
    if not url_or_path.startswith("http://") and not url_or_path.startswith("https://"):
        parts = url_or_path.split("/")
        if len(parts) >= 2:
            repo_id = "/".join(parts[:-1])
            fname = parts[-1]
            if not fname.endswith(".safetensors"):
                fname += ".safetensors"
            log_message(f"Downloading LoRA via Hugging Face Hub: {repo_id}/{fname}...")
            cached_path = hf_hub_download(repo_id=repo_id, filename=fname, token=hf_token)
            dest = LORAS_DIR / fname
            shutil.copy2(cached_path, dest)
            lora_id = re.sub(r'[^a-zA-Z0-9_]', '_', fname.replace('.safetensors', '')).lower()
            name = custom_name or fname.replace('.safetensors', '').replace('_', ' ')
            return {
                "id": lora_id,
                "name": name,
                "path": url_or_path,
                "filename": fname,
                "local_path": str(dest)
            }
        else:
            raise ValueError(f"Invalid LoRA repo format: {url_or_path}")

    # 2. Hugging Face Web URLs: https://huggingface.co/org/repo/blob/main/file.safetensors
    m_hf = re.match(r'https?://huggingface\.co/([^/]+)/([^/]+)/(?:blob|resolve)/([^/]+)/(.+)', url_or_path)
    if m_hf:
        user_org, repo_name, revision, subpath = m_hf.groups()
        repo_id = f"{user_org}/{repo_name}"
        log_message(f"Detected Hugging Face repo '{repo_id}', file '{subpath}'...")
        try:
            cached_path = hf_hub_download(repo_id=repo_id, filename=subpath, revision=revision, token=hf_token)
            fname = Path(subpath).name
            dest = LORAS_DIR / fname
            shutil.copy2(cached_path, dest)
            lora_id = re.sub(r'[^a-zA-Z0-9_]', '_', fname.replace('.safetensors', '')).lower()
            name = custom_name or fname.replace('.safetensors', '').replace('_', ' ')
            return {
                "id": lora_id,
                "name": name,
                "path": f"{repo_id}/{subpath}",
                "filename": fname,
                "local_path": str(dest)
            }
        except Exception as hf_err:
            log_message(f"hf_hub_download fallback to direct HTTP: {hf_err}")

    # 3. HTTP / HTTPS with browser session (Civitai, Cloudflare, Direct Link)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,application/octet-stream,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"
    })

    target_url = url_or_path
    inferred_name = ""
    if "civitai" in target_url:
        target_url, inferred_name = resolve_civitai_download_url(target_url, session, civitai_token)
        # Normalize civitai host to civitai.com
        target_url = re.sub(r'https?://civitai\.(red|green|blue|org)/', 'https://civitai.com/', target_url)
        if civitai_token and "token=" not in target_url:
            separator = "&" if "?" in target_url else "?"
            target_url = f"{target_url}{separator}token={civitai_token}"

    headers = {}
    if "huggingface.co" in target_url and hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    elif "civitai.com" in target_url and civitai_token:
        headers["Authorization"] = f"Bearer {civitai_token}"

    log_message(f"Connecting to download source: {target_url[:80]}...")
    resp = session.get(target_url, headers=headers, stream=True, allow_redirects=True, timeout=30)

    # Check for Civitai login redirect / auth block
    if resp.status_code in (401, 403) or (resp.history and any("/login" in r.headers.get("location", "") for r in resp.history)) or "/login" in resp.url:
        raise RuntimeError(
            "This Civitai LoRA requires authentication (Mature/NSFW content flag). "
            "Please provide your Civitai API Key in the box below, or upload the .safetensors file directly using the 'Upload from PC' tab."
        )

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP Error {resp.status_code} ({resp.reason}) from download URL. Check if the link is public or requires authentication.")

    # Extract filename from Content-Disposition header
    cd = resp.headers.get("content-disposition", "")
    fname = ""
    if cd:
        cd_match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\r\n]+)["\']?', cd, re.IGNORECASE)
        if cd_match:
            fname = unquote(cd_match.group(1)).strip()

    if not fname or not fname.endswith(".safetensors"):
        parsed_final = urlparse(resp.url)
        url_fname = unquote(Path(parsed_final.path).name)
        if url_fname and url_fname.endswith(".safetensors"):
            fname = url_fname

    if not fname:
        fname = f"{custom_name or inferred_name or 'custom_lora'}".replace(" ", "_")
        if not fname.endswith(".safetensors"):
            fname += ".safetensors"

    dest = LORAS_DIR / fname
    total_size = int(resp.headers.get("content-length", 0))
    downloaded = 0

    log_message(f"Downloading '{fname}' ({total_size / (1024*1024):.1f} MB)..." if total_size else f"Downloading '{fname}'...")
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    log_message(f"Downloaded '{fname}' successfully ({downloaded / (1024*1024):.1f} MB).")

    lora_id = re.sub(r'[^a-zA-Z0-9_]', '_', fname.replace('.safetensors', '')).lower()
    name = custom_name or inferred_name or fname.replace('.safetensors', '').replace('_', ' ')
    return {
        "id": lora_id,
        "name": name,
        "path": url_or_path,
        "filename": fname,
        "local_path": str(dest)
    }


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


class AddLoraRequest(BaseModel):
    name: Optional[str] = ""
    url: str
    scale: float = 1.0
    is_turbo: bool = False
    description: Optional[str] = ""
    civitai_token: Optional[str] = ""


class UpdateLoraRequest(BaseModel):
    id: str
    scale: Optional[float] = None
    enabled: Optional[bool] = None


class JobStartRequest(BaseModel):
    chapter_name: str = "chapter_01"
    max_dim: int = 1440
    num_steps: int = 8
    guidance_scale: float = 1.0
    prompt: str = "masterpiece, best quality, anime style, highly detailed coloring, vibrant color harmony, crisp lineart"
    negative_prompt: str = "worst quality, low quality, blurry, artifact, oversaturated, deformed, greyscale, monochrome"
    loras: List[LoraConfig] = []


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
        copied_images.append({
            "name": Path(src).name,
            "path": str(dst),
            "size": dst.stat().st_size,
            "url": f"/api/image/input/{Path(src).name}"
        })

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

    # Clear current input directory
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    temp_zip = BASE_DIR / f"temp_upload_{int(time.time())}.zip"
    try:
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        import zipfile
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        uploaded = []
        raw_stem = Path(file.filename).stem
        detected_chapter = raw_stem

        with zipfile.ZipFile(temp_zip, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename.replace("\\", "/"))
                if member_path.suffix.lower() in image_extensions:
                    base_name = member_path.name
                    # Ignore hidden / system files
                    if base_name.startswith("._") or base_name.startswith("."):
                        continue
                    dst = INPUT_DIR / base_name
                    with zf.open(member) as source, open(dst, "wb") as target:
                        shutil.copyfileobj(source, target)

                    uploaded.append({
                        "name": base_name,
                        "path": str(dst),
                        "size": dst.stat().st_size,
                        "url": f"/api/image/input/{base_name}"
                    })

        if not uploaded:
            raise HTTPException(status_code=400, detail="No valid image files found inside the archive.")

        def natural_sort_key(item):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', item["name"])]

        uploaded = sorted(uploaded, key=natural_sort_key)

        clean_ch = re.sub(r'[^\w\-_.]', '_', detected_chapter) or "chapter_01"
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

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        ext = Path(base_name).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue

        dst = INPUT_DIR / base_name
        with open(dst, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded.append({
            "name": base_name,
            "path": str(dst),
            "size": dst.stat().st_size,
            "url": f"/api/image/input/{base_name}"
        })

    # Sort images by natural filename order
    def natural_sort_key(item):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', item["name"])]

    uploaded = sorted(uploaded, key=natural_sort_key)

    with state_lock:
        if detected_chapter:
            job_state["chapter_name"] = re.sub(r'[^\w\-_.]', '_', detected_chapter)
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
    for f in sorted(glob.glob(str(INPUT_DIR / "*"))):
        ext = Path(f).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            images.append({
                "name": Path(f).name,
                "path": f,
                "size": os.path.getsize(f),
                "url": f"/api/image/input/{Path(f).name}"
            })
    return {"images": images, "chapter_name": job_state["chapter_name"]}


@app.post("/api/clear-files")
def clear_files():
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    with state_lock:
        job_state["input_images"] = []
        job_state["total_images"] = 0
    return {"status": "cleared"}


@app.get("/api/loras")
def get_loras():
    """Returns list of all configured LoRAs."""
    return {"loras": load_loras_config()}


@app.post("/api/loras/add")
async def add_lora_endpoint(req: AddLoraRequest):
    """Downloads a new LoRA from link / HF path and registers it in loras.json."""
    if not req.url:
        raise HTTPException(status_code=400, detail="Please provide a LoRA download link or Hugging Face path.")
    try:
        info = await asyncio.to_thread(download_lora_from_url, req.url, req.name or "", req.civitai_token or "")
        loras = load_loras_config()

        # Check if already exists
        for l in loras:
            if l.get("filename") == info["filename"] or l.get("id") == info["id"]:
                l["scale"] = req.scale
                l["enabled"] = True
                l["is_turbo"] = req.is_turbo
                if req.name:
                    l["name"] = req.name
                save_loras_config(loras)
                log_message(f"Updated existing LoRA: {l['name']}")
                return {"status": "success", "lora": l, "updated": True}

        new_lora = {
            "id": info["id"],
            "name": req.name or info["name"],
            "path": info["path"],
            "filename": info["filename"],
            "scale": req.scale,
            "enabled": True,
            "is_turbo": req.is_turbo,
            "is_custom": True,
            "description": req.description or f"Custom LoRA: {info['filename']}"
        }
        loras.append(new_lora)
        save_loras_config(loras)
        log_message(f"Successfully added LoRA: {new_lora['name']} ({info['filename']})")
        return {"status": "success", "lora": new_lora, "updated": False}
    except Exception as e:
        err_msg = str(e)
        log_message(f"Failed to add LoRA: {err_msg}")
        raise HTTPException(status_code=400, detail=f"Failed to download LoRA: {err_msg}")


@app.post("/api/loras/upload")
async def upload_lora_endpoint(
    file: UploadFile = File(...),
    name: Optional[str] = Form(""),
    scale: float = Form(1.0),
    is_turbo: bool = Form(False),
    description: Optional[str] = Form("")
):
    """Uploads a local .safetensors LoRA file directly from the user's PC."""
    if not file.filename.endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="Only .safetensors files are supported for LoRAs.")

    dest = LORAS_DIR / file.filename
    staging_dest = BASE_DIR / "kaggle_datasets" / "Loras_Manga_Coloring_Anima" / file.filename
    staging_dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        shutil.copy2(dest, staging_dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    lora_id = re.sub(r'[^a-zA-Z0-9_]', '_', file.filename.replace('.safetensors', '')).lower()
    lora_name = name or file.filename.replace('.safetensors', '').replace('_', ' ')

    loras = load_loras_config()
    for l in loras:
        if l.get("filename") == file.filename or l.get("id") == lora_id:
            l["scale"] = scale
            l["enabled"] = True
            l["is_turbo"] = is_turbo
            if name:
                l["name"] = name
            save_loras_config(loras)
            log_message(f"Uploaded and updated LoRA: {l['name']}")
            return {"status": "success", "lora": l, "updated": True}

    new_lora = {
        "id": lora_id,
        "name": lora_name,
        "path": file.filename,
        "filename": file.filename,
        "scale": scale,
        "enabled": True,
        "is_turbo": is_turbo,
        "is_custom": True,
        "description": description or f"Local LoRA: {file.filename}"
    }
    loras.append(new_lora)
    save_loras_config(loras)
    log_message(f"Successfully uploaded LoRA: {lora_name} ({file.filename})")
    return {"status": "success", "lora": new_lora, "updated": False}


@app.delete("/api/loras/{lora_id}")
def delete_lora_endpoint(lora_id: str):
    """Deletes a custom LoRA from loras.json and disk."""
    loras = load_loras_config()
    target = next((l for l in loras if l.get("id") == lora_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="LoRA not found")

    loras = [l for l in loras if l.get("id") != lora_id]
    save_loras_config(loras)

    # Delete local file if present in loras/ and staging
    if target.get("filename"):
        for d in [LORAS_DIR, BASE_DIR / "kaggle_datasets" / "Loras_Manga_Coloring_Anima"]:
            local_f = d / target["filename"]
            if local_f.exists():
                try:
                    local_f.unlink()
                except Exception:
                    pass

    log_message(f"Removed LoRA: {target.get('name')}")
    return {"status": "deleted", "id": lora_id}


@app.post("/api/loras/update")
def update_lora_endpoint(req: UpdateLoraRequest):
    """Updates scale and enabled properties of a LoRA in loras.json."""
    loras = load_loras_config()
    for l in loras:
        if l.get("id") == req.id:
            if req.scale is not None:
                l["scale"] = req.scale
            if req.enabled is not None:
                l["enabled"] = req.enabled
            save_loras_config(loras)
            return {"status": "success", "lora": l}
    raise HTTPException(status_code=404, detail="LoRA not found")


@app.get("/api/datasets/status")
async def get_datasets_status_endpoint():
    """Checks the status of Models_Manga_Coloring and Loras_Manga_Coloring_Anima on Kaggle."""
    from scripts.sync_kaggle_datasets import check_dataset_status
    u = get_kaggle_username()
    status = await asyncio.to_thread(check_dataset_status, u)
    return status


@app.post("/api/datasets/sync-loras")
def sync_loras_endpoint(background_tasks: BackgroundTasks):
    """Synchronizes all local LoRAs to Kaggle dataset in background."""
    from scripts.sync_kaggle_datasets import sync_loras_dataset
    u = get_kaggle_username()

    def _sync_worker():
        log_message(f"Starting Kaggle LoRAs dataset sync (Loras_Manga_Coloring_Anima) for @{u}...")
        try:
            sync_loras_dataset(u)
            log_message("Kaggle LoRAs dataset synchronized successfully!")
        except Exception as e:
            log_message(f"LoRA dataset sync error: {e}")

    background_tasks.add_task(_sync_worker)
    return {"status": "started", "message": "LoRAs synchronization started in background"}


@app.post("/api/datasets/sync-models")
def sync_models_endpoint(background_tasks: BackgroundTasks):
    """Synchronizes Anima Base model to Kaggle dataset in background."""
    from scripts.sync_kaggle_datasets import sync_models_dataset
    u = get_kaggle_username()

    def _sync_worker():
        log_message(f"Starting Kaggle Model dataset sync (Models_Manga_Coloring) for @{u}...")
        try:
            sync_models_dataset(u)
            log_message("Kaggle Model dataset synchronized successfully!")
        except Exception as e:
            log_message(f"Model dataset sync error: {e}")

    background_tasks.add_task(_sync_worker)
    return {"status": "started", "message": "Model synchronization started in background"}


class SaveSettingsRequest(BaseModel):
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    hf_token: Optional[str] = None


def get_kaggle_api_safe(verify_online: bool = False):
    """Lazily imports and tests KaggleApi without raising unhandled errors."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        if verify_online:
            u = getattr(api, "config_values", {}).get("username") or os.getenv("KAGGLE_USERNAME")
            api.dataset_list(user=u)
        return api
    except Exception:
        return None


def get_kaggle_username() -> str:
    """Dynamically resolves the authenticated Kaggle username."""
    api = get_kaggle_api_safe()
    if api:
        if hasattr(api, "config_values") and api.config_values.get("username"):
            return api.config_values["username"]
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

    return "mobasharulislam"


@app.get("/api/settings")
def get_settings_endpoint():
    """Returns the current Kaggle and Hugging Face authentication status."""
    api = get_kaggle_api_safe()
    is_authenticated = api is not None
    username = get_kaggle_username() if is_authenticated else None

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

    if key_or_token:
        # Clear legacy cached access tokens
        (kaggle_dir / "access_token").unlink(missing_ok=True)
        (kaggle_dir / "access_token.txt").unlink(missing_ok=True)
        (kaggle_dir / "token").unlink(missing_ok=True)

        if key_or_token.startswith("KGAT_"):
            (kaggle_dir / "access_token").write_text(key_or_token, encoding="utf-8")
            os.environ["KAGGLE_API_TOKEN"] = key_or_token
        
        creds = {
            "username": username or "rabiulislam5788",
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


def background_tpu_runner(req: JobStartRequest):
    """Executes the full pipeline: Notebook Build -> Kaggle Push -> Monitor -> Download."""
    sys_py = sys.executable
    kaggle_user = get_kaggle_username()
    try:
        with state_lock:
            job_state["status"] = "preparing"
            job_state["phase_message"] = "Generating TPU multi-core notebook..."
            job_state["start_time"] = time.time()
            job_state["error"] = None
            job_state["completed_images"] = 0
            job_state["logs"] = []

        log_message("=== Starting Manga Colorization Job ===")
        log_message(f"User: @{kaggle_user} | Chapter: {req.chapter_name} | Max Dim: {req.max_dim}px | Steps: {req.num_steps}")

        # 1. If local images or zips exist in INPUT_DIR, package them into a fast single zip and upload to Kaggle dataset
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        local_scans = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in image_extensions]
        local_zips = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() == ".zip"]

        if local_scans or local_zips:
            with state_lock:
                job_state["phase_message"] = f"Packaging & uploading scans to Kaggle..."
            log_message(f"Preparing single fast zip archive for Kaggle dataset '{kaggle_user}/gbd-ch-107-part2'...")

            staging_dir = BASE_DIR / "kaggle_datasets" / "input_pages_staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            for old_f in staging_dir.iterdir():
                try:
                    if old_f.is_file():
                        old_f.unlink()
                except Exception:
                    pass

            zip_dest = staging_dir / "scans.zip"
            if local_scans:
                import zipfile
                log_message(f"Compressing {len(local_scans)} scans into {zip_dest.name}...")
                with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for scan in sorted(local_scans):
                        zf.write(scan, arcname=scan.name)
            elif local_zips:
                import shutil
                shutil.copy2(local_zips[0], zip_dest)

            dataset_slug = "gbd-ch-107-part2"
            dataset_full_slug = f"{kaggle_user}/{dataset_slug}"

            meta = {
                "title": dataset_slug,
                "id": dataset_full_slug,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(staging_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

            api = get_kaggle_api_safe()
            if not api:
                raise RuntimeError("Kaggle API is not authenticated. Please check your Kaggle API key in Settings.")

            # Check if dataset exists on Kaggle under this user
            existing_datasets = []
            try:
                existing_datasets = [getattr(d, "ref", "") or getattr(d, "id", "") for d in api.dataset_list(user=kaggle_user)]
            except Exception as e:
                log_message(f"Note checking existing datasets: {e}")

            dataset_exists = any(dataset_full_slug.lower() == d.lower() or dataset_slug.lower() == d.split("/")[-1].lower() for d in existing_datasets)

            if dataset_exists:
                try:
                    log_message(f"Updating existing Kaggle dataset '{dataset_full_slug}'...")
                    api.dataset_create_version(str(staging_dir), version_notes=f"Chapter: {req.chapter_name}", dir_mode="zip")
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
            
            # Poll until Kaggle marks dataset version as 'ready' (preventing kernel mounting stale previous version)
            dataset_full_slug = f"{kaggle_user}/gbd-ch-107-part2"
            max_wait_seconds = 600  # 10 minutes max for massive datasets
            poll_interval = 4
            waited = 0
            while waited < max_wait_seconds:
                time.sleep(poll_interval)
                waited += poll_interval
                try:
                    ds_status = str(api.dataset_status(dataset_full_slug)).lower()
                    if "ready" in ds_status:
                        log_message(f"Kaggle dataset '{dataset_full_slug}' is ready! ({waited}s)")
                        break
                    elif "error" in ds_status or "fail" in ds_status:
                        raise RuntimeError(f"Kaggle dataset processing failed: {ds_status}")
                    else:
                        if waited % 12 == 0:
                            log_message(f"Dataset indexing on Kaggle... ({waited}s, status: {ds_status})")
                            with state_lock:
                                job_state["phase_message"] = f"Kaggle indexing dataset ({waited}s)..."
                except Exception as poll_err:
                    if "processing failed" in str(poll_err):
                        raise
                    # Ignore transient status API errors during initial upload window
                    pass
            log_message("Kaggle dataset version confirmed ready.")

        # 2. Build lightweight notebook with user selected LoRAs & settings
        from scripts.make_notebook import build_notebook
        lora_dicts = [lora.dict() for lora in req.loras] if req.loras else None
        build_notebook(
            input_dir=str(INPUT_DIR),
            loras=lora_dicts,
            max_dim=req.max_dim,
            num_steps=req.num_steps,
            guidance_scale=req.guidance_scale,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            out_notebook_path="kernel_anima/tpu_image_gen_anima.ipynb"
        )
        log_message("Notebook successfully compiled.")

        # 3. Push to Kaggle
        with state_lock:
            job_state["status"] = "pushing"
            job_state["phase_message"] = "Pushing kernel to Kaggle TPU Cluster..."
        log_message("Pushing kernel to Kaggle...")

        push_res = subprocess.run([sys_py, "-m", "kaggle", "kernels", "push", "-p", "kernel_anima"], capture_output=True, text=True, cwd=BASE_DIR)
        log_message(push_res.stdout.strip())
        if push_res.returncode != 0:
            raise RuntimeError(f"Kaggle push failed: {push_res.stderr}")

        # 3. Monitor Kaggle Kernel execution
        kernel_name = f"{kaggle_user}/tpu-image-generation-anima-base-1-0-2b"
        is_running = False

        while True:
            time.sleep(15)
            status_res = subprocess.run([sys_py, "-m", "kaggle", "kernels", "status", kernel_name], capture_output=True, text=True, cwd=BASE_DIR)
            status_text = status_res.stdout.strip()
            
            with state_lock:
                job_state["elapsed_seconds"] = int(time.time() - job_state["start_time"])
                job_state["last_kernel_status"] = status_text

            if "QUEUED" in status_text:
                with state_lock:
                    job_state["status"] = "queued"
                    job_state["phase_message"] = "Queued on Kaggle TPU allocator..."
                log_message(f"Status: QUEUED ({job_state['elapsed_seconds']}s elapsed)")
            elif "RUNNING" in status_text:
                if not is_running:
                    is_running = True
                    log_message("TPU VM Allocated! 8 Cores are initializing & colorizing...")
                with state_lock:
                    job_state["status"] = "running"
                    job_state["phase_message"] = "8 TPU Cores Generating in Parallel..."
                log_message(f"Status: RUNNING ({job_state['elapsed_seconds']}s elapsed)")
            elif "COMPLETE" in status_text:
                log_message("Kaggle Kernel Completed Successfully!")
                break
            elif "ERROR" in status_text or "CANCELLED" in status_text:
                raise RuntimeError(f"Kaggle kernel failed with status: {status_text}")

        # 4. Download Results
        safe_chapter = re.sub(r'[^\w\-_.]', '_', req.chapter_name)
        chapter_out = OUTPUT_DIR / safe_chapter
        chapter_out.mkdir(parents=True, exist_ok=True)

        with state_lock:
            job_state["status"] = "downloading"
            job_state["phase_message"] = "Downloading finished high-res color images..."
        log_message(f"Downloading outputs to {chapter_out}...")

        dl_res = subprocess.run([sys_py, str(SCRIPTS_DIR / "download_fixed.py"), str(chapter_out), kaggle_user], capture_output=True, text=True, cwd=BASE_DIR)
        log_message(dl_res.stdout[-300:] if dl_res.stdout else "Downloaded successfully.")

        # 5. Populate Output Results
        generated_dir = chapter_out / "generated"
        out_files = []
        if generated_dir.exists():
            for f in sorted(glob.glob(str(generated_dir / "*.png"))):
                out_files.append({
                    "name": Path(f).name,
                    "path": f,
                    "url": f"/api/image/output/{safe_chapter}/{Path(f).name}"
                })

        with state_lock:
            job_state["status"] = "completed"
            job_state["phase_message"] = "Colorization Complete!"
            job_state["completed_images"] = len(out_files)
            job_state["output_images"] = out_files

        log_message(f"SUCCESS: Generated {len(out_files)} colorized pages in {job_state['elapsed_seconds']}s total!")

    except Exception as e:
        err_msg = str(e)
        log_message(f"ERROR: {err_msg}")
        with state_lock:
            job_state["status"] = "error"
            job_state["phase_message"] = "Colorization Failed"
            job_state["error"] = err_msg


@app.post("/api/start")
def start_colorization_job(req: JobStartRequest, background_tasks: BackgroundTasks):
    with state_lock:
        if job_state["status"] in ["preparing", "pushing", "queued", "running", "downloading"]:
            raise HTTPException(status_code=400, detail="A colorization job is already in progress.")

    background_tasks.add_task(background_tpu_runner, req)
    return {"status": "started", "chapter": req.chapter_name}


@app.get("/api/status")
def get_job_status():
    with state_lock:
        if job_state["status"] in ["preparing", "pushing", "queued", "running", "downloading"]:
            job_state["elapsed_seconds"] = int(time.time() - job_state["start_time"])
        return dict(job_state)


@app.get("/api/results")
def get_results(chapter: Optional[str] = None):
    # Discover all available chapters in output_app
    available_chapters = []
    if OUTPUT_DIR.exists():
        for ch_dir in OUTPUT_DIR.iterdir():
            if ch_dir.is_dir() and (ch_dir / "generated").exists() and list((ch_dir / "generated").glob("*.png")):
                available_chapters.append(ch_dir.name)

    # Sort chapters naturally
    available_chapters.sort(key=lambda c: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', c)])

    # Pick requested chapter, or current job chapter, or first available chapter
    selected_chapter = chapter
    if not selected_chapter:
        curr_job_ch = job_state.get("chapter_name")
        if curr_job_ch and curr_job_ch in available_chapters:
            selected_chapter = curr_job_ch
        elif available_chapters:
            selected_chapter = available_chapters[0]
        else:
            selected_chapter = "Chapter_1"

    safe_chapter = re.sub(r'[^\w\-_.]', '_', selected_chapter)
    generated_dir = OUTPUT_DIR / safe_chapter / "generated"

    pairs = []
    if generated_dir.exists():
        input_files = sorted(glob.glob(str(INPUT_DIR / "*")), key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', Path(p).name)])
        input_map = {}
        for p in input_files:
            p_path = Path(p)
            if p_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                input_map[p_path.stem] = p_path.name
                num_match = re.search(r'\d+', p_path.stem)
                if num_match:
                    num_str = num_match.group(0)
                    input_map[num_str.lstrip("0") or "0"] = p_path.name
                    input_map[f"{int(num_str):02d}"] = p_path.name
        
        color_files = sorted(glob.glob(str(generated_dir / "*.png")), key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', Path(p).name)])

        for idx, color_file in enumerate(color_files):
            color_filename = Path(color_file).name
            stem = Path(color_file).stem
            clean_stem = stem.replace("edit_", "").replace("color_", "").replace("out_", "")

            input_url = None
            if clean_stem in input_map:
                input_url = f"/api/image/input/{input_map[clean_stem]}"
            elif stem in input_map:
                input_url = f"/api/image/input/{input_map[stem]}"
            else:
                num_match = re.search(r'\d+', clean_stem)
                if num_match and (num_match.group(0).lstrip("0") or "0") in input_map:
                    input_url = f"/api/image/input/{input_map[num_match.group(0).lstrip('0') or '0']}"
                else:
                    for in_stem, in_name in input_map.items():
                        if in_stem == clean_stem or in_stem.startswith(clean_stem) or clean_stem.startswith(in_stem):
                            input_url = f"/api/image/input/{in_name}"
                            break
                if not input_url and idx < len(input_files):
                    input_url = f"/api/image/input/{Path(input_files[idx]).name}"

            pairs.append({
                "id": clean_stem,
                "color_name": color_filename,
                "color_url": f"/api/image/output/{safe_chapter}/{color_filename}",
                "input_url": input_url or f"/api/image/output/{safe_chapter}/{color_filename}",
            })

    return {
        "chapter_name": safe_chapter,
        "available_chapters": available_chapters,
        "count": len(pairs),
        "pairs": pairs
    }


@app.get("/api/image/input/{filename:path}")
def get_input_image(filename: str):
    base_name = Path(filename).name
    file_path = INPUT_DIR / base_name
    if not file_path.exists():
        file_path = INPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Input image not found")
    return FileResponse(str(file_path))


@app.get("/api/image/output/{chapter}/{filename}")
def get_output_image(chapter: str, filename: str):
    safe_chapter = re.sub(r'[^\w\-_.]', '_', chapter)
    file_path = OUTPUT_DIR / safe_chapter / "generated" / filename
    if not file_path.exists():
        # Fallback check in out_v31
        alt_path = BASE_DIR / "output_anima" / safe_chapter / "generated" / filename
        if alt_path.exists():
            return FileResponse(str(alt_path))
        raise HTTPException(status_code=404, detail="Output image not found")
    return FileResponse(str(file_path))


@app.post("/api/open-folder")
def open_output_folder(chapter_name: Optional[str] = None):
    chapter = chapter_name or job_state.get("chapter_name", "chapter_01")
    safe_chapter = re.sub(r'[^\w\-_.]', '_', str(chapter))
    target_dir = OUTPUT_DIR / safe_chapter / "generated"
    if not target_dir.exists():
        target_dir = OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        os.startfile(str(target_dir))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(target_dir)])
    else:
        subprocess.run(["xdg-open", str(target_dir)])
    return {"status": "opened", "path": str(target_dir)}


# Mount static web directory
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
