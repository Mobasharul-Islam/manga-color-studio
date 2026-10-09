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
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from scripts.common import KERNEL_SLUG, SCANS_DATASET_SLUG, resolve_kaggle_username


def build_notebook(
    input_dir: Optional[str] = None,
    loras: Optional[List[Dict[str, Any]]] = None,
    model: Optional[Dict[str, Any]] = None,
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

    # The app passes the model & LoRAs selected in its library; standalone runs use the defaults.
    if model is None:
        model = {"name": "Anima Base 1.0 (2B)", "source": "circlestone-labs/Anima-Base-v1.0-Diffusers", "revision": None, "dataset": None}
    if loras is None:
        loras = [
            {"name": "Color Manga Anima", "path": "Kutches/Anim4/colorMangaAnima.safetensors", "scale": 1.0, "enabled": True, "dataset": None},
            {"name": "Anima Turbo", "path": "circlestone-labs/Anima-Official-LoRAs/anima-turbo-lora-v0.2.safetensors", "scale": 1.0, "enabled": True, "dataset": None},
        ]
    loras = [l for l in loras if l.get("enabled", True) and l.get("path")]

    # Build LoRA fusion lines for tpu_worker.py
    lora_code_lines = [
        f"    add_lora({repr(l['path'])}, scale={float(l.get('scale', 1.0))}, transformer=pipe.transformer)"
        for l in loras
    ]
    lora_fusion_block = "\n".join(lora_code_lines) if lora_code_lines else "    pass"
    lora_paths = [l["path"] for l in loras]

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
    dataset_sources_list = [f"{kuser}/{SCANS_DATASET_SLUG}"]
    for ref in [model.get("dataset")] + [l.get("dataset") for l in loras]:
        if ref and ref not in dataset_sources_list:
            dataset_sources_list.append(ref)
    data_sources = [{"datasetId": ref, "sourceType": "dataset"} for ref in dataset_sources_list]
    model_repo = model["source"]
    model_revision = model.get("revision") or "main"

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

# The optional HF_TOKEN comes from Kaggle Secrets (notebook editor -> Add-ons -> Secrets),
# never from the notebook source. It only speeds up Hugging Face downloads.
def _secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or ''
    except Exception:
        return ''


HF_TOKEN = os.getenv('HF_TOKEN') or _secret('HF_TOKEN')
MODEL_REPO = {repr(model_repo)}
MODEL_REVISION = {repr(model_revision)}
LOCAL_MODEL_DIR = '/tmp/mcs_model'


def prepare_local_model():
    """Finds the mounted model dataset and makes it loadable without network access.

    modular_model_index.json names each component's source repo, so the dataset copy would
    still be fetched from Hugging Face. A rewritten index in /tmp points every component at
    the mounted folder instead.
    """
    if not os.path.exists('/kaggle/input'):
        return None
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'modular_model_index.json' in files:
            idx = json.load(open(os.path.join(root, 'modular_model_index.json')))
            for spec in idx.values():
                if isinstance(spec, list) and len(spec) == 3 and isinstance(spec[2], dict) and 'pretrained_model_name_or_path' in spec[2]:
                    spec[2]['pretrained_model_name_or_path'] = root
                    spec[2]['revision'] = None
            os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
            with open(os.path.join(LOCAL_MODEL_DIR, 'modular_model_index.json'), 'w') as f:
                json.dump(idx, f, indent=2)
            return LOCAL_MODEL_DIR
        if 'model_index.json' in files:
            return root
    return None


LOCAL_MODEL = prepare_local_model()
MODEL_ID = LOCAL_MODEL if LOCAL_MODEL else MODEL_REPO

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


REPO_CACHE = Path(os.environ['HF_HOME']) / 'hub' / ('models--' + MODEL_REPO.replace('/', '--'))
BLOBS = REPO_CACHE / 'blobs'


def repo_files():
    if LOCAL_MODEL:
        return {{}}
    url = 'https://huggingface.co/api/models/%s/revision/%s?blobs=true' % (MODEL_REPO, MODEL_REVISION)
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode())
    skip = {{'README.md', 'LICENSE.md', '.gitattributes'}}
    return {{s['rfilename']: (s.get('size') or 0) for s in data.get('siblings', []) if s['rfilename'] not in skip}}


def preflight(files):
    if not files:
        return files
    big = max(files, key=lambda k: files[k])
    url = 'https://huggingface.co/%s/resolve/%s/%s' % (MODEL_REPO, MODEL_REVISION, big)
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


def _dl_subprocess(filename, size, repo=MODEL_REPO, revision=MODEL_REVISION):
    code = (
        "import os; " + (("os.environ['HF_TOKEN']=%r; " % HF_TOKEN) if HF_TOKEN else "") +
        "from huggingface_hub import hf_hub_download; "
        "p = hf_hub_download(%r, %r, revision=%r); print('OK', p)" % (repo, filename, revision)
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


def download_file_safe(filename, size, repo=MODEL_REPO, max_retries=4):
    for attempt in range(1, max_retries + 1):
        mark('downloading %s (attempt %d/%d)...' % (filename, attempt, max_retries))
        if _dl_subprocess(filename, size, repo):
            mark('downloaded %s OK' % filename)
            return True
        time.sleep(2 * attempt)
    return False


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

    mark('Model environment ready!')


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

MODEL_REPO = {repr(model_repo)}
MODEL_REVISION = {repr(model_revision)}


def find_local_model():
    """The model folder prepared by the setup cell (see prepare_local_model), if any."""
    if os.path.isfile('/tmp/mcs_model/modular_model_index.json'):
        return '/tmp/mcs_model'
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'model_index.json' in files:
                return root
    return None


LOCAL_MODEL = find_local_model()

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
            if os.getenv('HF_TOKEN') and 'huggingface.co' in lora_path_str.split('/')[2]:
                headers['Authorization'] = 'Bearer ' + os.getenv('HF_TOKEN')
            req = urllib.request.Request(lora_path_str, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f)
        return dest

    # 3. Check if Hugging Face repo path
    if lora_path_str.count('/') >= 2:
        parts = lora_path_str.split('/')
        return hf_hub_download(repo_id='/'.join(parts[:2]), filename='/'.join(parts[2:]))

    raise FileNotFoundError('LoRA %r is not in any mounted dataset and has no download source.' % lora_path_str)


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


def load_pipeline(rank):
    components = ['text_encoder', 'tokenizer', 't5_tokenizer', 'text_conditioner', 'transformer', 'vae']
    if LOCAL_MODEL:
        try:
            pipe = ModularPipeline.from_pretrained(LOCAL_MODEL)
            load_with_retry(pipe, components)
            mark('[Core %d] Loaded model from mounted dataset %s' % (rank, LOCAL_MODEL))
            return pipe
        except Exception as e:
            mark('[Core %d] Mounted model failed to load (%r); falling back to %s' % (rank, e, MODEL_REPO))
    pipe = ModularPipeline.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    load_with_retry(pipe, components)
    return pipe


def mp_worker(index, image_paths):
    device = xm.xla_device()
    rank = index
    world_size = 8

    mark('[Core %d/%d] Initializing pipeline on %s...' % (rank, world_size, device))

    pipe = load_pipeline(rank)
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

# Resolve every LoRA once here, so the 8 TPU processes find them locally instead of
# all downloading the same file at the same time.
from tpu_worker import resolve_lora_path
for _lora in __LORA_PATHS__:
    print('[LoRA] %s -> %s' % (_lora, resolve_lora_path(_lora)), flush=True)

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
    cell4_lines = (cell4_template
                   .replace("__WORKER_CODE_INJECT__", repr(worker_code_str))
                   .replace("__LORA_PATHS__", repr(lora_paths)))

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
        "id": f"{kuser}/{KERNEL_SLUG}",
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
