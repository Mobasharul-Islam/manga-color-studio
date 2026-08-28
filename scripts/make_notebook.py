import os
import io
import sys
import glob
import json
import base64
import tarfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Root paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def build_notebook(
    input_dir: Optional[str] = None,
    loras: Optional[List[Dict[str, Any]]] = None,
    max_dim: int = 1440,
    num_steps: int = 8,
    guidance_scale: float = 1.0,
    prompt: str = "",
    negative_prompt: str = "",
    out_notebook_path: str = "kernel_anima/tpu_image_gen_anima.ipynb"
):
    max_dim = min(int(max_dim), 1440)
    prompt = prompt.strip() if (prompt and prompt.strip()) else "masterpiece, best quality, anime style, highly detailed coloring, vibrant color harmony, crisp lineart"
    negative_prompt = negative_prompt.strip() if (negative_prompt and negative_prompt.strip()) else "worst quality, low quality, blurry, artifact, oversaturated, deformed, greyscale, monochrome"
    cells = []

    # Cell 0: Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# TPU v5-8 8-Core Manga Colorization Studio\n",
            "\n",
            "- **8-Way Multiprocessing (`xmp.spawn`)**: Worker module `tpu_worker.py` executes on all 8 TPU chips.\n",
            "- **Native Resolution (`MAX_DIM = %d`)**: Generates full scan resolution.\n" % max_dim,
            "- **Direct LoRA Fusion (`add_lora`)**: Zero PEFT overhead in-place tensor fusion.\n",
            "- **Cosmos Temporal Reference Hook**: Preserves original lineart and panel fidelity.\n",
        ]
    })

    # Cell 1: Package Installations
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip uninstall -y -q google-tunix 2>/dev/null; pip install -q --upgrade \"transformers==5.15.0\" \"tokenizers==0.22.2\" accelerate safetensors peft kaggle\n",
            "!pip install --upgrade \"huggingface_hub @ git+https://github.com/huggingface/huggingface_hub.git@f34cd32cac1f3603e3294543e1fb0eb4508d5139\" \"diffusers @ git+https://github.com/huggingface/diffusers.git@102c24934fe1705e6caaadb03789b0f2542fd8a9\"\n",
        ]
    })

    # Cell 2: Imports & Environment Unlocking
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "# Remove Kaggle single-host env vars to unlock 8-core distributed multiprocessing\n",
            "os.environ.pop('TPU_PROCESS_ADDRESSES', None)\n",
            "os.environ.pop('CLOUD_TPU_TASK_ID', None)\n",
            "\n",
            "import huggingface_hub\n",
            "try:\n",
            "    from huggingface_hub import get_cached_repo_tree  # noqa: F401\n",
            "except ImportError:\n",
            "    raise RuntimeError(\n",
            "        'Installed huggingface_hub is too old for diffusers main ('\n",
            "        + huggingface_hub.__version__\n",
            "        + '); re-run cell 1 (install) or restart the session and Run All.'\n",
            "    )\n",
            "import torch\n",
            "import torch_xla\n",
            "import torch_xla.core.xla_model as xm\n",
            "import torch_xla.distributed.xla_multiprocessing as xmp\n",
            "import diffusers\n",
            "import transformers\n",
            "import tokenizers\n",
            "import safetensors\n",
            "from diffusers import ModularPipeline\n",
            "\n",
            "print('huggingface_hub:', huggingface_hub.__version__)\n",
            "print('torch:', torch.__version__)\n",
            "print('torch_xla:', torch_xla.__version__)\n",
            "print('diffusers:', diffusers.__version__)\n",
            "print('PJRT_DEVICE:', os.getenv('PJRT_DEVICE'))\n",
        ]
    })

    # Build LoRA fusion lines for tpu_worker.py
    if loras is None:
        loras_file = BASE_DIR / "loras.json"
        if loras_file.exists():
            try:
                with open(loras_file, "r", encoding="utf-8") as f:
                    loras = json.load(f).get("loras", [])
            except Exception:
                loras = None
        if not loras:
            loras = [
                {"name": "Color Manga Anima", "path": "Kutches/Anim4/colorMangaAnima.safetensors", "scale": 1.0, "enabled": True},
                {"name": "Anima Turbo", "path": "circlestone-labs/Anima-Official-LoRAs/anima-turbo-lora-v0.2.safetensors", "scale": 1.0, "enabled": True}
            ]

    lora_code_lines = []
    for lora in loras:
        if not lora.get("enabled", True):
            continue
        p = lora.get("path", "")
        s = float(lora.get("scale", 1.0))
        if p:
            lora_code_lines.append(f"    add_lora({repr(p)}, scale={s}, transformer=pipe.transformer)")

    lora_fusion_block = "\n".join(lora_code_lines) if lora_code_lines else "    pass"

    # Determine active Kaggle credentials for cloud auto-caching
    kuser = None
    kkey = ""
    ktoken = ""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        if hasattr(api, "config_values") and api.config_values.get("username"):
            kuser = api.config_values.get("username")
            kkey = api.config_values.get("key") or ""
            ktoken = api.config_values.get("api_token") or api.config_values.get("token") or ""
        elif api.get_config_value("username"):
            kuser = api.get_config_value("username")
            kkey = api.get_config_value("key") or ""
    except Exception:
        pass

    if not ktoken:
        ktoken = os.getenv("KAGGLE_API_TOKEN", "")
        if not ktoken:
            acc_tok_path = Path.home() / ".kaggle" / "access_token"
            if acc_tok_path.exists():
                try:
                    ktoken = acc_tok_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

    if not kuser or kuser == "your_kaggle_username":
        u = os.getenv("KAGGLE_USERNAME")
        if u and u != "your_kaggle_username":
            kuser = u
        kkey = kkey or os.getenv("KAGGLE_KEY", "")

    if not kuser or not kkey:
        try:
            kjson = Path.home() / ".kaggle" / "kaggle.json"
            if kjson.exists():
                kdata = json.loads(kjson.read_text(encoding="utf-8"))
                kuser = kuser or kdata.get("username")
                kkey = kkey or kdata.get("key", "")
        except Exception:
            pass

    kuser = kuser or "mobasharulislam"
    kkey = kkey or ""
    ktoken = ktoken or ""
    hf_token_val = os.getenv("HF_TOKEN", "")

    # Resolve dataset IDs
    scan_dataset_id = f"{kuser}/gbd-ch-107-part2"
    data_sources = [{"datasetId": scan_dataset_id, "sourceType": "dataset"}]
    dataset_sources_list = [scan_dataset_id]

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        user_ds = [getattr(d, "ref", "") or getattr(d, "id", "") for d in api.dataset_list(user=kuser)]
        
        if any("models-manga-coloring" in d.lower() for d in user_ds):
            user_model_id = f"{kuser}/models-manga-coloring"
            data_sources.append({"datasetId": user_model_id, "sourceType": "dataset"})
            dataset_sources_list.append(user_model_id)
            
        if any("loras-manga-coloring-anima" in d.lower() for d in user_ds):
            user_lora_id = f"{kuser}/loras-manga-coloring-anima"
            data_sources.append({"datasetId": user_lora_id, "sourceType": "dataset"})
            dataset_sources_list.append(user_lora_id)
    except Exception:
        pass

    # Cell 3: Pre-download & Cache
    cell3_lines = f'''import faulthandler
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

faulthandler.enable()

# Setup Kaggle credentials inside Kaggle notebook for cloud auto-caching
try:
    kdir = Path('/root/.kaggle')
    kdir.mkdir(parents=True, exist_ok=True)
    if {repr(ktoken)}:
        (kdir / 'access_token').write_text({repr(ktoken)}, encoding='utf-8')
        try:
            os.chmod(str(kdir / 'access_token'), 0o600)
        except Exception:
            pass
        os.environ['KAGGLE_API_TOKEN'] = {repr(ktoken)}
    if {repr(kuser)} and {repr(kkey)}:
        kjson = {{"username": {repr(kuser)}, "key": {repr(kkey)}}}
        (kdir / 'kaggle.json').write_text(json.dumps(kjson, indent=2), encoding='utf-8')
        try:
            os.chmod(str(kdir / 'kaggle.json'), 0o600)
        except Exception:
            pass
        os.environ['KAGGLE_USERNAME'] = {repr(kuser)}
        os.environ['KAGGLE_KEY'] = {repr(kkey)}
except Exception:
    pass

def find_local_model():
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'model_index.json' in files:
                return root
    return None

LOCAL_MODEL = find_local_model()
MODEL_ID = LOCAL_MODEL if LOCAL_MODEL else 'circlestone-labs/Anima-Base-v1.0-Diffusers'
MANGA_LORA_REPO = 'Kutches/Anim4'
MANGA_LORA_FILE = 'colorMangaAnima.safetensors'
TURBO_LORA_REPO = 'circlestone-labs/Anima-Official-LoRAs'
TURBO_LORA_FILE = 'anima-turbo-lora-v0.2.safetensors'
HF_TOKEN = {repr(hf_token_val)} or os.getenv('HF_TOKEN', '')

if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '120'

WORKING = Path('/kaggle/working')
WORKING.mkdir(exist_ok=True)
DEBUG_LOG = WORKING / 'debug.log'

T0 = time.time()
STAGE = ['starting']


def _open_log():
    f = open(DEBUG_LOG, 'a', buffering=1)
    f.write('=== session %s ===\\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
    return f


_LOG = _open_log()


def mark(stage):
    line = '[%7.1fs] %s' % (time.time() - T0, stage)
    print(line, flush=True)
    _LOG.write(line + '\\n')
    _LOG.flush()


REPO_CACHE = Path(os.environ['HF_HOME']) / 'hub' / ('models--' + ('circlestone-labs--Anima-Base-v1.0-Diffusers' if not LOCAL_MODEL else 'local'))
BLOBS = REPO_CACHE / 'blobs'


def repo_files():
    if LOCAL_MODEL:
        return {{}}
    url = 'https://huggingface.co/api/models/%s?blobs=true' % MODEL_ID
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode())
    skip = {{'README.md', 'LICENSE.md', '.gitattributes'}}
    return {{s['rfilename']: (s.get('size') or 0) for s in data.get('siblings', []) if s['rfilename'] not in skip}}


def preflight(files):
    if not files:
        return files
    big = max(files, key=lambda k: files[k])
    url = 'https://huggingface.co/%s/resolve/main/%s' % (MODEL_ID, big)
    req = urllib.request.Request(url, headers={{'User-Agent': 'Mozilla/5.0',
                                               'Range': 'bytes=0-%d' % (8 * 1024 * 1024 - 1)}})
    t0 = time.time()
    got = 0
    with urllib.request.urlopen(req, timeout=60) as r:
        host = r.geturl().split('/')[2]
        while got < 8 * 1024 * 1024:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            got += len(chunk)
    dt = max(time.time() - t0, 1e-6)
    mark('preflight: classic HTTP to %s (host %s) = %.2f MB/s' % (big, host, got / 1e6 / dt))
    return files


def _cached(filename):
    snaps = REPO_CACHE / 'snapshots'
    if not snaps.is_dir():
        return False
    for d in snaps.iterdir():
        if (d / filename).exists():
            return True
    return False


def _incomplete_bytes():
    total = 0
    dirs = []
    if BLOBS.is_dir():
        dirs.append(BLOBS)
    xet_cache = Path(os.environ['HF_HOME']) / 'xet'
    if xet_cache.is_dir():
        dirs.append(xet_cache)
    for d in dirs:
        for p in d.rglob('*'):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return total


def _dl_subprocess(filename, size, repo=MODEL_ID):
    code = (
        "import os; " + (("os.environ['HF_TOKEN']=%r; " % HF_TOKEN) if HF_TOKEN else "") +
        "from huggingface_hub import hf_hub_download; "
        "p = hf_hub_download(%r, %r); print('OK', p)" % (repo, filename)
    )
    if size < 50 * 1024 * 1024:
        timeout = 300
    else:
        timeout = max(600, int(size / (500 * 1024)))

    t0 = time.time()
    prev_bytes = _incomplete_bytes()
    last_prog = t0
    proc = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while proc.poll() is None:
        time.sleep(5)
        now = time.time()
        cur_bytes = _incomplete_bytes()
        if cur_bytes > prev_bytes:
            prev_bytes = cur_bytes
            last_prog = now
        elif now - last_prog > 180:
            proc.kill()
            mark('subprocess download STALLED for >180s on %s, killed' % filename)
            return False
        if now - t0 > timeout:
            proc.kill()
            mark('subprocess download TIMEOUT (>%ds) on %s, killed' % (timeout, filename))
            return False

    out, err = proc.communicate()
    ok = (proc.returncode == 0) and (b'OK' in out)
    if not ok:
        mark('subprocess download FAIL on %s (code %d): %s' % (filename, proc.returncode, err.decode(errors='replace')[:400]))
    return ok


def download_file_safe(filename, size, repo=MODEL_ID, max_retries=4):
    for attempt in range(1, max_retries + 1):
        mark('downloading %s (attempt %d/%d)...' % (filename, attempt, max_retries))
        if _dl_subprocess(filename, size, repo):
            mark('downloaded %s OK' % filename)
            return True
        time.sleep(2 * attempt)
    return False


def auto_cache_to_kaggle_datasets():
    """Background task: Uploads downloaded model & LoRAs directly to Kaggle Datasets for 0-second boot on future runs."""
    try:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=False)
            from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        u = getattr(api, "config_values", {{}}).get("username") or getattr(api, "_config_values", {{}}).get("username") or os.getenv("KAGGLE_USERNAME") or {repr(kuser)}
        if not u:
            return

        user_datasets = [getattr(d, "ref", "") or getattr(d, "id", "") for d in api.dataset_list(user=u)]
        import shutil

        # 1. Cache Anima Base model
        model_slug = f"{{u}}/models-manga-coloring"
        if not any("models-manga-coloring" in d.lower() for d in user_datasets):
            snaps = REPO_CACHE / 'snapshots'
            if snaps.is_dir():
                latest_snap = next(iter(snaps.iterdir()), None)
                if latest_snap and latest_snap.is_dir():
                    mark('[Cloud-Cache] Packaging Anima Base model into private Kaggle dataset (%s)...' % model_slug)
                    model_staging = Path('/kaggle/working/model_staging')
                    if model_staging.exists():
                        shutil.rmtree(model_staging, ignore_errors=True)
                    shutil.copytree(latest_snap, model_staging, symlinks=False)

                    meta = {{
                        "title": "Models_Manga_Coloring",
                        "id": model_slug,
                        "licenses": [{{"name": "CC0-1.0"}}]
                    }}
                    with open(model_staging / 'dataset-metadata.json', 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=2)

                    api.dataset_create_new(
                        str(model_staging),
                        dir_mode="zip",
                        public=False,
                        quiet=False
                    )
                    shutil.rmtree(model_staging, ignore_errors=True)
                    mark('[Cloud-Cache] Successfully created %s on Kaggle! Future runs will boot in 0 seconds.' % model_slug)

        # 2. Cache LoRAs
        lora_slug = f"{{u}}/loras-manga-coloring-anima"
        if not any("loras-manga-coloring-anima" in d.lower() for d in user_datasets):
            lora_staging = Path('/kaggle/working/loras_staging')
            if lora_staging.exists():
                shutil.rmtree(lora_staging, ignore_errors=True)
            lora_staging.mkdir(parents=True, exist_ok=True)

            for p in Path(os.environ['HF_HOME']).rglob('*.safetensors'):
                shutil.copy2(p, lora_staging / p.name)

            lora_files = list(lora_staging.glob('*.safetensors'))
            if lora_files:
                mark('[Cloud-Cache] Packaging LoRAs into private Kaggle dataset (%s)...' % lora_slug)
                meta = {{
                    "title": "Loras_Manga_Coloring_Anima",
                    "id": lora_slug,
                    "licenses": [{{"name": "CC0-1.0"}}]
                }}
                with open(lora_staging / 'dataset-metadata.json', 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)

                api.dataset_create_new(
                    str(lora_staging),
                    dir_mode="zip",
                    public=False,
                    quiet=False
                )
                shutil.rmtree(lora_staging, ignore_errors=True)
                mark('[Cloud-Cache] Successfully created %s on Kaggle! Future runs will mount LoRAs in 0 seconds.' % lora_slug)
    except Exception as cache_err:
        mark('[Cloud-Cache Note] %s' % cache_err)


def warm_cache():
    if LOCAL_MODEL:
        mark('Zero-wait boot: Model detected locally at %s! Skipping 5.6GB HF download.' % LOCAL_MODEL)
    else:
        mark('warming Hugging Face cache for Anima Base...')
        files = repo_files()
        preflight(files)

        small = [f for f, s in files.items() if s < 50 * 1024 * 1024]
        large = [f for f, s in files.items() if s >= 50 * 1024 * 1024]

        for f in small:
            if not _cached(f):
                download_file_safe(f, files[f])

        for f in large:
            if not _cached(f):
                download_file_safe(f, files[f])

    mark('pre-caching LoRAs if needed...')
    # Check if LoRAs are in any mounted dataset
    ds_has_loras = False
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if MANGA_LORA_FILE in files:
                ds_has_loras = True
                break
    if ds_has_loras:
        mark('Zero-wait boot: LoRAs mounted in dataset! Skipping LoRA download.')
    else:
        download_file_safe(MANGA_LORA_FILE, 100 * 1024 * 1024, repo=MANGA_LORA_REPO)
        download_file_safe(TURBO_LORA_FILE, 100 * 1024 * 1024, repo=TURBO_LORA_REPO)
    mark('Model & LoRA environment ready!')

    if not LOCAL_MODEL:
        threading.Thread(target=auto_cache_to_kaggle_datasets, daemon=True).start()


warm_cache()
'''
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell3_lines.splitlines()]
    })

    # Cell 4: 8-Core Worker & Execution
    worker_code_str = f'''import os
import sys
import gc
import re
import time
import math
import glob
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import diffusers
from diffusers import ModularPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def find_local_model():
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'model_index.json' in files:
                return root
    return None

LOCAL_MODEL = find_local_model()
MODEL_ID = LOCAL_MODEL if LOCAL_MODEL else 'circlestone-labs/Anima-Base-v1.0-Diffusers'
MANGA_LORA_REPO = 'Kutches/Anim4'
MANGA_LORA_FILE = 'colorMangaAnima.safetensors'
TURBO_LORA_REPO = 'circlestone-labs/Anima-Official-LoRAs'
TURBO_LORA_FILE = 'anima-turbo-lora-v0.2.safetensors'

MAX_DIM = {max_dim}
NUM_STEPS = {num_steps}
GUIDANCE_SCALE = {guidance_scale}
PROMPT = {repr(prompt)}
NEGATIVE_PROMPT = {repr(negative_prompt)}

T0 = time.time()


def mark(msg):
    print('[%7.1fs] %s' % (time.time() - T0, msg), flush=True)


def mem_gb():
    try:
        return xm.get_memory_info(xm.xla_device())
    except Exception:
        return 'N/A'


def pad_to_aspect(img, max_dim=1440):
    orig_w, orig_h = img.size
    canvas_h = max(32, int(round(max_dim / 32.0)) * 32)
    canvas_w = max(32, int(round((max_dim * (1024.0 / 1440.0)) / 32.0)) * 32)

    scale = min(canvas_w / orig_w, canvas_h / orig_h)
    scaled_w = max(1, min(canvas_w, int(round(orig_w * scale))))
    scaled_h = max(1, min(canvas_h, int(round(orig_h * scale))))

    scaled_img = img.resize((scaled_w, scaled_h), Image.LANCZOS)
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))

    pad_x = (canvas_w - scaled_w) // 2
    pad_y = (canvas_h - scaled_h) // 2
    canvas.paste(scaled_img, (pad_x, pad_y))
    crop_box = (pad_x, pad_y, pad_x + scaled_w, pad_y + scaled_h)
    return canvas, crop_box


_KOHYA_TO_DIFFUSERS = {{
    'self_attn_q_proj':       'attn1.to_q',
    'self_attn_k_proj':       'attn1.to_k',
    'self_attn_v_proj':       'attn1.to_v',
    'self_attn_output_proj':  'attn1.to_out.0',
    'cross_attn_q_proj':      'attn2.to_q',
    'cross_attn_k_proj':      'attn2.to_k',
    'cross_attn_v_proj':      'attn2.to_v',
    'cross_attn_output_proj': 'attn2.to_out.0',
    'mlp_layer1':             'ff.net.0.proj',
    'mlp_layer2':             'ff.net.2',
}}

_TURBO_MAP = {{
    'self_attn.q_proj': 'attn1.to_q',
    'self_attn.k_proj': 'attn1.to_k',
    'self_attn.v_proj': 'attn1.to_v',
    'self_attn.output_proj': 'attn1.to_out.0',
    'cross_attn.q_proj': 'attn2.to_q',
    'cross_attn.k_proj': 'attn2.to_k',
    'cross_attn.v_proj': 'attn2.to_v',
    'cross_attn.output_proj': 'attn2.to_out.0',
    'mlp.layer1': 'ff.net.0.proj',
    'mlp.layer2': 'ff.net.2',
}}


def extract_lora_deltas(sd, scale=1.0):
    weights_A, weights_B, alphas = {{}}, {{}}, {{}}
    is_kohya = any(k.startswith('lora_unet_blocks_') for k in sd.keys())

    if is_kohya:
        for key, value in sd.items():
            if not key.startswith('lora_unet_blocks_'):
                continue
            parts = key.split('.')
            tokens = parts[0][len('lora_unet_blocks_'):].split('_')
            block_idx = tokens[0]
            submod = '_'.join(tokens[1:])
            diffusers_submod = _KOHYA_TO_DIFFUSERS.get(submod)
            if diffusers_submod is None:
                continue
            target_name = 'transformer_blocks.%s.%s.weight' % (block_idx, diffusers_submod)
            rest = parts[1:]
            if rest == ['alpha']:
                alphas[target_name] = value.item() if hasattr(value, 'item') else float(value)
            elif len(rest) == 2 and rest[1] == 'weight':
                if rest[0] == 'lora_down':
                    weights_A[target_name] = value
                elif rest[0] == 'lora_up':
                    weights_B[target_name] = value
    else:
        for k, v in sd.items():
            if k.startswith('diffusion_model.blocks.'):
                m = re.match(r'diffusion_model\\.blocks\\.(\\d+)\\.(.*?)\\.(lora_[AB]\\.weight|alpha)', k)
                if m:
                    block_idx, submod, param_type = m.groups()
                    diff_sub = _TURBO_MAP.get(submod)
                    if diff_sub:
                        target_name = 'transformer_blocks.%s.%s.weight' % (block_idx, diff_sub)
                        if param_type == 'alpha':
                            alphas[target_name] = v.item() if hasattr(v, 'item') else float(v)
                        elif 'lora_A' in param_type:
                            weights_A[target_name] = v
                        elif 'lora_B' in param_type:
                            weights_B[target_name] = v

    deltas = {{}}
    for name in weights_A:
        if name in weights_B:
            A = weights_A[name].float()
            B = weights_B[name].float()
            rank = A.shape[0]
            alpha = alphas.get(name, float(rank))
            deltas[name] = (B @ A) * ((alpha / rank) * scale)
    return deltas


def resolve_lora_path(lora_path_str):
    if os.path.isfile(lora_path_str):
        return lora_path_str

    filename = os.path.basename(lora_path_str)
    if '?' in filename:
        filename = filename.split('?')[0]

    # 1. Check mounted Kaggle datasets & local working folders
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if filename in files:
                return os.path.join(root, filename)
    for d in ['/kaggle/working/loras', '/kaggle/working']:
        cand = os.path.join(d, filename)
        if os.path.isfile(cand):
            return cand

    # 2. Check if direct HTTP/HTTPS URL
    if lora_path_str.startswith('http://') or lora_path_str.startswith('https://'):
        os.makedirs('/kaggle/working/loras', exist_ok=True)
        dest = os.path.join('/kaggle/working/loras', filename)
        if not os.path.exists(dest):
            mark('Downloading LoRA from URL: %s' % lora_path_str)
            import urllib.request, shutil
            headers = {{'User-Agent': 'Mozilla/5.0'}}
            if os.getenv('HF_TOKEN'):
                headers['Authorization'] = 'Bearer ' + os.getenv('HF_TOKEN')
            req = urllib.request.Request(lora_path_str, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f)
        return dest

    # 3. Check if Hugging Face repo path
    if '/' in lora_path_str:
        parts = lora_path_str.split('/')
        if len(parts) == 3:
            repo_id = parts[0] + '/' + parts[1]
            fname = parts[2]
            return hf_hub_download(repo_id=repo_id, filename=fname)
        elif len(parts) == 2 and parts[1].endswith('.safetensors'):
            repo_id = parts[0]
            fname = parts[1]
            return hf_hub_download(repo_id=repo_id, filename=fname)

    return lora_path_str


def add_lora(lora_path, scale=1.0, transformer=None):
    file_path = resolve_lora_path(str(lora_path))
    mark('Fusing LoRA: %s (scale=%.2f)...' % (os.path.basename(file_path), scale))
    sd = load_file(file_path)
    deltas = extract_lora_deltas(sd, scale=scale)

    target_model = transformer if transformer is not None else getattr(globals().get('pipe', None), 'transformer', None)
    if target_model is None:
        raise ValueError('No transformer specified or found to fuse LoRA into.')

    fused_count = 0
    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if name in deltas:
                d = deltas[name].to(device=param.device, dtype=param.dtype)
                if param.shape == d.shape:
                    param.add_(d)
                    fused_count += 1

    mark('  Successfully fused %d parameter matrices from %s | mem: %s' % (fused_count, os.path.basename(file_path), mem_gb()))
    return fused_count


def install_cosmos_reference_hook(pipe):
    orig_forward = pipe.transformer.forward

    def reference_forward(hidden_states, timestep, encoder_hidden_states=None, **kwargs):
        ref_latent = getattr(pipe.transformer, '_ref_latent', None)
        if ref_latent is not None:
            is_4d = (hidden_states.ndim == 4)
            if is_4d:
                hidden_states = hidden_states.unsqueeze(2)

            hidden_states_5d = torch.cat([hidden_states, ref_latent], dim=2)
            out = orig_forward(hidden_states=hidden_states_5d, timestep=timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)

            if hasattr(out, 'sample'):
                sample = out.sample[:, :, :1, :, :]
                if is_4d:
                    sample = sample.squeeze(2)
                out.sample = sample
                return out
            elif isinstance(out, (tuple, list)):
                sample = out[0][:, :, :1, :, :]
                if is_4d:
                    sample = sample.squeeze(2)
                return (sample,)
        else:
            return orig_forward(hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)

    pipe.transformer.forward = reference_forward


def install_step_progress(pipe, rank):
    sched = getattr(pipe, 'scheduler', None)
    if sched is None or not hasattr(sched, 'step'):
        return None
    orig_step = sched.step

    def step(*args, **kwargs):
        out = orig_step(*args, **kwargs)
        try:
            xm.mark_step()
        except Exception:
            pass
        return out

    sched.step = step


def load_with_retry(pipe, components, max_attempts=3):
    components = [components] if isinstance(components, str) else list(components)
    for attempt in range(1, max_attempts + 1):
        pipe.load_components(dtype=torch.bfloat16)
        missing = [c for c in components if getattr(pipe, c, None) is None]
        if not missing:
            return
        time.sleep(10)
    raise RuntimeError('Failed to load components %r' % components)


def mp_worker(index, image_paths):
    device = xm.xla_device()
    rank = index
    world_size = 8

    mark('[Core %d/%d] Initializing pipeline on %s...' % (rank, world_size, device))

    pipe = ModularPipeline.from_pretrained(MODEL_ID)
    load_with_retry(pipe, ['text_encoder', 'tokenizer', 't5_tokenizer', 'text_conditioner', 'transformer', 'vae'])
    pipe = pipe.to(device)
    torch_xla.sync(wait=True)

    # Attach add_lora helper
    pipe.add_lora = lambda path, scale=1.0: add_lora(path, scale=scale, transformer=pipe.transformer)

    # Fuse selected LoRAs
{lora_fusion_block}
    torch_xla.sync(wait=True)

    install_cosmos_reference_hook(pipe)
    install_step_progress(pipe, rank)

    out_dir = '/kaggle/working/generated'
    os.makedirs(out_dir, exist_ok=True)

    # Shard dataset evenly across all TPU cores
    my_paths = image_paths[rank::world_size]
    mark('[Core %d/%d] Ready! Assigned %d images | memory: %s' % (rank, world_size, len(my_paths), mem_gb()))

    _t0 = time.time()
    for i, path in enumerate(my_paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, 'edit_%s.png' % stem)
        if os.path.exists(out_path):
            mark('[Core %d] [%d/%d] skip (already exists): %s' % (rank, i + 1, len(my_paths), out_path))
            continue
        ref_latent = None
        ref_tensor = None
        result = None
        try:
            _t1 = time.time()
            orig_img = Image.open(path).convert('RGB')
            orig_size = orig_img.size
            canvas_pil, crop_box = pad_to_aspect(orig_img, MAX_DIM)
            w, h = canvas_pil.size
            mark('[Core %d] [%d/%d] %s -> generating (%d, %d, orig %s)...' % (rank, i + 1, len(my_paths), os.path.basename(path), w, h, orig_size))

            arr = np.array(canvas_pil).astype(np.float32) / 127.5 - 1.0
            ref_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

            with torch.no_grad():
                enc = pipe.vae.encode(ref_tensor)
                if hasattr(enc, 'latent_dist'):
                    ref_latent = enc.latent_dist.sample()
                elif isinstance(enc, tuple):
                    ref_latent = enc[0]
                else:
                    ref_latent = getattr(enc, 'latents', enc)

                if ref_latent.ndim == 4:
                    ref_latent = ref_latent.unsqueeze(2)

                if hasattr(pipe.vae.config, 'latents_mean') and pipe.vae.config.latents_mean is not None:
                    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=ref_latent.device, dtype=ref_latent.dtype).view(1, -1, 1, 1, 1)
                    latents_std = torch.tensor(pipe.vae.config.latents_std, device=ref_latent.device, dtype=ref_latent.dtype).view(1, -1, 1, 1, 1)
                    ref_latent = (ref_latent - latents_mean) / latents_std
                elif hasattr(pipe.vae.config, 'scaling_factor') and pipe.vae.config.scaling_factor is not None:
                    ref_latent = ref_latent * pipe.vae.config.scaling_factor

            pipe.transformer._ref_latent = ref_latent

            try:
                xm.mark_step()
            except Exception:
                pass
            gc.collect()

            pipe_kwargs = {{
                'prompt': [PROMPT],
                'num_inference_steps': NUM_STEPS,
                'guidance_scale': GUIDANCE_SCALE,
                'height': h,
                'width': w,
            }}
            if NEGATIVE_PROMPT and GUIDANCE_SCALE > 1.0:
                pipe_kwargs['negative_prompt'] = [NEGATIVE_PROMPT]

            result = pipe(**pipe_kwargs)
            try:
                xm.mark_step()
            except Exception:
                pass

            colorized = result.images[0]
            colorized_crop = colorized.crop(crop_box)
            if colorized_crop.size != orig_size:
                colorized_crop = colorized_crop.resize(orig_size, Image.LANCZOS)

            colorized_crop.save(out_path)

            dt = int(time.time() - _t1)
            mark('[Core %d] [%d/%d] DONE %s -> %s (%ds, res %s, mem %s)'
                 % (rank, i + 1, len(my_paths), os.path.basename(path), out_path, dt, colorized_crop.size, mem_gb()))
        except Exception as e:
            traceback.print_exc()
            mark('[Core %d] [%d/%d] FAILED %s: %r' % (rank, i + 1, len(my_paths), os.path.basename(path), e))
            try:
                torch_xla.sync(wait=False)
            except Exception:
                pass
            try:
                xm.mark_step()
            except Exception:
                pass
        finally:
            pipe.transformer._ref_latent = None
            del ref_tensor, ref_latent, result
            gc.collect()

    try:
        xm.mark_step()
    except Exception:
        pass
    mark('[Core %d] ALL %d IMAGES FINISHED in %.0fs' % (rank, len(my_paths), time.time() - _t0))
'''

    cell4_template = """worker_code = __WORKER_CODE_INJECT__

with open('/kaggle/working/tpu_worker.py', 'w', encoding='utf-8') as f:
    f.write(worker_code)

if '/kaggle/working' not in sys.path:
    sys.path.insert(0, '/kaggle/working')

import glob
import os
import sys
import time
from PIL import Image
import torch_xla.distributed.xla_multiprocessing as xmp
from tpu_worker import mp_worker, add_lora

# ----------------------------------------------------
# Auto-Unzip: Check for scan zip archives across all mounted datasets in /kaggle/input
# ----------------------------------------------------
import zipfile

UNZIP_DIR = '/kaggle/working/input_pages'
os.makedirs(UNZIP_DIR, exist_ok=True)

if os.path.exists('/kaggle/input'):
    for root, dirs, files in os.walk('/kaggle/input'):
        for zf in files:
            if zf.lower().endswith('.zip'):
                if 'model' in zf.lower():
                    continue
                zf_path = os.path.join(root, zf)
                print(f"[Auto-Unzip] Found scan archive '{zf}' at '{zf_path}'. Extracting to {UNZIP_DIR}...", flush=True)
                try:
                    with zipfile.ZipFile(zf_path, 'r') as zip_ref:
                        zip_ref.extractall(UNZIP_DIR)
                    print(f"[Auto-Unzip] Successfully extracted '{zf}'!", flush=True)
                except Exception as ze:
                    print(f"[Auto-Unzip Warning] Failed to extract {zf_path}: {ze}", flush=True)

IMAGE_PATHS = []
image_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

# 1. Check UNZIP_DIR first
for root, dirs, files in os.walk(UNZIP_DIR):
    for f in sorted(files):
        if f.lower().endswith(image_exts) and not f.startswith('.'):
            IMAGE_PATHS.append(os.path.join(root, f))

# 2. Check all /kaggle/input (excluding model/lora folders) if UNZIP_DIR had no images
if not IMAGE_PATHS and os.path.exists('/kaggle/input'):
    for root, dirs, files in os.walk('/kaggle/input'):
        lower_root = root.lower()
        if 'model' in lower_root or 'lora' in lower_root:
            continue
        for f in sorted(files):
            if f.lower().endswith(image_exts) and not f.startswith('.'):
                IMAGE_PATHS.append(os.path.join(root, f))

if not IMAGE_PATHS:
    print("[Diagnostic] Full directory tree of /kaggle/input:", flush=True)
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            print(f"  {root} -> {files}", flush=True)
    else:
        print("  /kaggle/input does not exist!", flush=True)
    print(f"[Diagnostic] Files in /kaggle/working: {os.listdir('/kaggle/working')}", flush=True)
    raise FileNotFoundError('No manga scans found in /kaggle/input or /kaggle/working/input_pages!')

IMAGE_PATHS = sorted(IMAGE_PATHS)
DATASET_DIR = os.path.dirname(IMAGE_PATHS[0])
print('Dataset selected: %s (%d raw images found)' % (DATASET_DIR, len(IMAGE_PATHS)), flush=True)

# ----------------------------------------------------
# Auto-Split: Detect double-spread landscape pages & split into 2 portrait halves
# Prevents expensive TPU XLA JIT shape re-compilation (~240s savings per spread!)
# ----------------------------------------------------
SPLIT_DIR = '/kaggle/working/split_pages'
os.makedirs(SPLIT_DIR, exist_ok=True)

WORK_QUEUE = []
SPREAD_MAP = {}

for p in IMAGE_PATHS:
    try:
        with Image.open(p) as img:
            w, h = img.size
            if w > h * 1.15:
                stem = os.path.splitext(os.path.basename(p))[0]
                left_crop = img.crop((0, 0, w // 2, h))
                right_crop = img.crop((w // 2, 0, w, h))

                left_path = os.path.join(SPLIT_DIR, f"{stem}__spread_left.png")
                right_path = os.path.join(SPLIT_DIR, f"{stem}__spread_right.png")

                left_crop.save(left_path)
                right_crop.save(right_path)

                WORK_QUEUE.append(left_path)
                WORK_QUEUE.append(right_path)

                SPREAD_MAP[stem] = {
                    "left_stem": f"{stem}__spread_left",
                    "right_stem": f"{stem}__spread_right",
                    "orig_path": p,
                    "orig_size": (w, h)
                }
                print(f"[Auto-Split] Detected double-spread '{os.path.basename(p)}' ({w}x{h}). Split into 2 portrait halves ({w//2}x{h}).", flush=True)
            else:
                WORK_QUEUE.append(p)
    except Exception as e:
        print(f"Error inspecting image '{p}': {e}", flush=True)
        WORK_QUEUE.append(p)

print('Prepared work queue: %d total portrait pages (including %d split double-spreads).' % (len(WORK_QUEUE), len(SPREAD_MAP)), flush=True)

print('Launching 8-Core Data-Parallel Colorization via PJRT...', flush=True)
_t_start = time.time()
try:
    xmp.spawn(mp_worker, args=(WORK_QUEUE,), nprocs=None)
except Exception as e:
    print('[Warning] Multi-core worker run completed with note: %s' % e, flush=True)
print('ALL 8 TPU CORES COMPLETED in %.0fs total!' % (time.time() - _t_start), flush=True)

# ----------------------------------------------------
# Auto-Stitch: Seamlessly stitch colored spread halves back to original dimensions
# ----------------------------------------------------
if SPREAD_MAP:
    print('Stitching %d double-spread pages...' % len(SPREAD_MAP), flush=True)
    out_dir = '/kaggle/working/generated'
    for stem, info in SPREAD_MAP.items():
        left_out = os.path.join(out_dir, f"edit_{info['left_stem']}.png")
        right_out = os.path.join(out_dir, f"edit_{info['right_stem']}.png")
        final_out = os.path.join(out_dir, f"edit_{stem}.png")

        if os.path.exists(left_out) and os.path.exists(right_out):
            try:
                left_img = Image.open(left_out).convert('RGB')
                right_img = Image.open(right_out).convert('RGB')

                target_h = max(left_img.height, right_img.height)
                if left_img.height != target_h:
                    left_img = left_img.resize((int(left_img.width * target_h / left_img.height), target_h), Image.LANCZOS)
                if right_img.height != target_h:
                    right_img = right_img.resize((int(right_img.width * target_h / right_img.height), target_h), Image.LANCZOS)

                stitched_w = left_img.width + right_img.width
                stitched = Image.new('RGB', (stitched_w, target_h))
                stitched.paste(left_img, (0, 0))
                stitched.paste(right_img, (left_img.width, 0))

                orig_w, orig_h = info['orig_size']
                if stitched.size != (orig_w, orig_h):
                    stitched = stitched.resize((orig_w, orig_h), Image.LANCZOS)

                stitched.save(final_out)
                print(f"[Auto-Stitch] Successfully stitched '{os.path.basename(final_out)}' ({orig_w}x{orig_h})!", flush=True)

                # Clean up split temporary files
                try:
                    os.remove(left_out)
                    os.remove(right_out)
                except Exception:
                    pass
            except Exception as err:
                print(f"[Auto-Stitch Error] Failed stitching '{stem}': {err}", flush=True)
        else:
            print(f"[Auto-Stitch Warning] Missing parts for '{stem}': left={os.path.exists(left_out)}, right={os.path.exists(right_out)}", flush=True)

# ----------------------------------------------------
# Package outputs into a single fast ZIP archive and clean up loose files
# ----------------------------------------------------
import shutil
import zipfile

out_dir = '/kaggle/working/generated'
zip_out_path = '/kaggle/working/generated.zip'

if os.path.isdir(out_dir):
    print(f"[Packaging] Compressing generated images into {zip_out_path}...", flush=True)
    with zipfile.ZipFile(zip_out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for file in sorted(files):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, '/kaggle/working')
                zf.write(full_path, arcname=rel_path)
    
    # Remove the loose generated folder and temp folders so Kaggle only outputs the single zip and log
    shutil.rmtree(out_dir, ignore_errors=True)

# Clean up temporary folders and files from /kaggle/working so Kaggle API doesn't download them individually
for temp_path in ['/kaggle/working/split_pages', '/kaggle/working/input_pages', '/kaggle/working/tpu_worker.py', '/kaggle/working/loras']:
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path, ignore_errors=True)
    elif os.path.isfile(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

print("All outputs packaged into /kaggle/working/generated.zip! Ready for fast single-archive download.", flush=True)
"""
    cell4_lines = cell4_template.replace("__WORKER_CODE_INJECT__", repr(worker_code_str))

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell4_lines.splitlines()]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "tpuV3-8",
                "dataSources": data_sources,
                "dockerImageVersionId": 30787,
                "isGpuEnabled": False,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = BASE_DIR / out_notebook_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    kernel_meta = {
        "id": f"{kuser}/tpu-image-generation-anima-base-1-0-2b",
        "title": "TPU Image Generation Anima Base 1.0 2B",
        "code_file": "tpu_image_gen_anima.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "true",
        "enable_internet": "true",
        "dataset_sources": dataset_sources_list,
        "kernel_sources": []
    }
    (out_file.parent / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=1), encoding="utf-8")

    print("Successfully generated %s and kernel-metadata.json for @%s" % (out_file, kuser))
    return str(out_file)


if __name__ == "__main__":
    build_notebook(input_dir=str(BASE_DIR / "input_pages"))
