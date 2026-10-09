"""Library worker: runs inside a small CPU-only Kaggle notebook (no TPU quota, no TPU queue).

Downloads one model or LoRA and publishes it as its own private Kaggle dataset, so TPU runs
can mount it instead of downloading it. app/library.py prepends `TASK = {...}` to this file
when it builds the notebook; nothing secret is ever embedded. Credentials come from the
notebook's Kaggle Secrets: KAGGLE_KEY (or KAGGLE_API_TOKEN) is required to create datasets,
HF_TOKEN and CIVITAI_TOKEN are optional for gated downloads.

The outcome is written to /kaggle/working/result.json, which the app downloads afterwards.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TASK = globals().get("TASK") or {}
RESULT = Path("/kaggle/working/result.json")
STAGE = Path("/tmp/mcs_stage")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or ""
    except Exception:
        return ""


def load_credentials():
    for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN", "HF_TOKEN", "CIVITAI_TOKEN"):
        val = os.getenv(name) or secret(name)
        if val:
            os.environ[name] = val
    os.environ.setdefault("KAGGLE_USERNAME", TASK["owner"])
    if not (os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN")):
        raise RuntimeError(
            "This notebook needs a KAGGLE_KEY (or KAGGLE_API_TOKEN) secret to save datasets. "
            "Open %s, go to Add-ons -> Secrets, attach the secret, then retry from the app." % TASK["notebook_url"]
        )


def filename_from_disposition(cd):
    m = re.search(r"filename\*?=(?:UTF-8'')?[\"']?([^\"';\r\n]+)", cd or "", re.IGNORECASE)
    return urllib.parse.unquote(m.group(1)).strip() if m else ""


def download_url(url, dest_dir, filename=""):
    headers = {"User-Agent": "Mozilla/5.0"}
    host = urllib.parse.urlparse(url).netloc.lower()
    if "civitai" in host and os.getenv("CIVITAI_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["CIVITAI_TOKEN"]
    if host.endswith("huggingface.co") and os.getenv("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403) and "civitai" in host:
            raise RuntimeError("Civitai refused the download (HTTP %d). Mature LoRAs need a CIVITAI_TOKEN secret "
                               "on this notebook, or add the LoRA in the app with your Civitai key." % e.code)
        raise
    with resp:
        if "/login" in resp.geturl():
            raise RuntimeError("Civitai redirected to its login page. Add a CIVITAI_TOKEN secret to this notebook.")
        name = filename or filename_from_disposition(resp.headers.get("content-disposition"))
        name = name or Path(urllib.parse.urlparse(resp.geturl()).path).name
        dest = dest_dir / Path(name).name
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
    return dest


def fetch_lora():
    from huggingface_hub import hf_hub_download

    src = TASK["source"]
    token = os.getenv("HF_TOKEN") or None
    m = re.match(r"https?://huggingface\.co/([^/]+)/([^/]+)/(?:blob|resolve)/([^/]+)/(.+)", src)
    if m:
        org, repo, revision, sub = m.groups()
        path = Path(hf_hub_download(org + "/" + repo, sub, revision=revision, token=token))
    elif not src.startswith(("http://", "https://")):
        parts = src.split("/")
        path = Path(hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]), token=token))
    else:
        path = download_url(TASK.get("download_url") or src, STAGE, TASK.get("filename", ""))

    dest = STAGE / (TASK.get("filename") or path.name)
    if path != dest:
        shutil.copy2(path, dest)
    if dest.suffix != ".safetensors":
        raise RuntimeError("Downloaded file '%s' is not a .safetensors LoRA." % dest.name)

    from safetensors import safe_open
    with safe_open(str(dest), framework="pt") as f:
        keys = list(f.keys())
    if not any(k.startswith(("lora_unet_blocks_", "diffusion_model.blocks.")) for k in keys):
        raise RuntimeError("'%s' does not look like an Anima LoRA (unrecognised weight names)." % dest.name)
    return dest.name


def fetch_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        TASK["source"],
        revision=TASK.get("revision") or None,
        local_dir=str(STAGE),
        token=os.getenv("HF_TOKEN") or None,
        ignore_patterns=["*.md", ".gitattributes", "*.png", "*.jpg", "*.jpeg", "*.webp"],
    )
    shutil.rmtree(STAGE / ".cache", ignore_errors=True)
    if not ((STAGE / "modular_model_index.json").exists() or (STAGE / "model_index.json").exists()):
        raise RuntimeError("The download has no model_index.json / modular_model_index.json.")
    return ""


def publish(filename):
    from kaggle.api.kaggle_api_extended import KaggleApi

    ref = TASK["dataset"]
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    entry = dict(TASK["library"], filename=filename or None, size_bytes=size,
                 added=time.strftime("%Y-%m-%d %H:%M:%S"))
    (STAGE / "library.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    (STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": TASK["title"], "id": ref, "licenses": [{"name": "unknown"}],
    }), encoding="utf-8")

    api = KaggleApi()
    api.authenticate()
    slug = ref.split("/", 1)[1]
    existing = [d.ref.lower() for d in (api.dataset_list(mine=True, search=slug) or []) if d]
    log("Uploading %.1f MB to private dataset %s..." % (size / 1e6, ref))
    if ref.lower() in existing:
        res = api.dataset_create_version(str(STAGE), version_notes="Re-cached from %s" % TASK["source"], dir_mode="zip")
    else:
        res = api.dataset_create_new(str(STAGE), public=False, dir_mode="zip")
    err = getattr(res, "error", None)
    if err:
        raise RuntimeError("Kaggle rejected the dataset: %s" % err)
    return size


def main():
    t0 = time.time()
    try:
        load_credentials()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle", "huggingface_hub", "safetensors"], check=False)
        shutil.rmtree(STAGE, ignore_errors=True)
        STAGE.mkdir(parents=True)
        log("Fetching %s %s..." % (TASK["kind"], TASK["source"]))
        filename = fetch_model() if TASK["kind"] == "model" else fetch_lora()
        size = publish(filename)
        result = {"ok": True, "dataset": TASK["dataset"], "filename": filename or None, "size_bytes": size}
        log("Done in %.0fs." % (time.time() - t0))
    except Exception as e:
        traceback.print_exc()
        result = {"ok": False, "error": str(e)}
    finally:
        shutil.rmtree(STAGE, ignore_errors=True)
    RESULT.write_text(json.dumps(dict(result, item_id=TASK.get("item_id")), indent=2), encoding="utf-8")


main()
