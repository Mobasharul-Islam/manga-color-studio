"""Colorization worker that runs inside the Kaggle notebook, on TPU or GPU.

scripts/make_notebook.py writes this file into the notebook with `CONFIG = {...}` prepended
(the run's settings: backend, model, LoRAs, prompt...). The notebook calls `main()`, which:

1. finds the cached model / LoRAs in the mounted datasets (or downloads them),
2. collects the scans (splitting double-page spreads on TPU, see below),
3. starts one process per TPU core or GPU, each colorizing its share of the pages,
4. stitches spreads back together and zips the results for download.

Backends:
- "tpu": 8 TPU cores via torch_xla, bfloat16. Every new image shape costs a ~4 minute XLA
  compile, so all pages share one fixed portrait canvas and spreads are split in two.
- "gpu": one process per CUDA GPU (2 on Kaggle's T4 x2). T4/P100 have no bfloat16, so the
  pipeline runs in float16 and is rebuilt in float32 if a page comes out blank (overflow).
  Any shape is cheap, so each page keeps its own aspect ratio and spreads stay whole.
- "cpu": local testing only.

Nothing secret is embedded: an optional HF_TOKEN is read from Kaggle Secrets.

Encryption (CONFIG["encrypted"]): the scans arrive as an AES-encrypted .mcsenc archive and the
results leave as one. The key comes from the MCS_ENCRYPTION_KEY Kaggle Secret and is checked
against the app's key fingerprint. Every readable page lives only in /tmp, never in
/kaggle/working (which Kaggle saves as the notebook's output), so even a crash mid-run leaves
nothing readable behind.
"""
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

# Must be set before CUDA initialises: lets PyTorch grow memory segments instead of failing
# on fragmentation when a T4 is nearly full.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

CONFIG = globals().get("CONFIG") or {}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
COMPONENTS = ["text_encoder", "tokenizer", "t5_tokenizer", "text_conditioner", "transformer", "vae"]
T0 = time.time()


def cfg(key, default=None):
    return CONFIG.get(key, default)


def input_root():
    return cfg("input_root", "/kaggle/input")


def work_dir():
    return cfg("work_dir", "/kaggle/working")


def encrypted():
    return bool(cfg("encrypted"))


def pages_dir():
    """Where scans and colorized pages live during the run. When encrypting, a folder outside
    /kaggle/working so nothing readable can end up in the saved notebook output."""
    return cfg("scratch_dir", "/tmp/mcs_work") if encrypted() else work_dir()


_KEY = {"value": None}


def load_key():
    """The passphrase from the app's temporary key dataset, checked against the app's fingerprint."""
    import mcs_crypto
    key = ""
    for root, _dirs, files in os.walk(input_root()):
        if mcs_crypto.KEY_FILE in files:
            with open(os.path.join(root, mcs_crypto.KEY_FILE), encoding="utf-8") as f:
                key = f.read().strip()
            break
    if not key:
        raise RuntimeError(
            "Encryption is on, but the temporary key dataset is not mounted in this notebook. "
            "Start the run again from the app.")
    if mcs_crypto.fingerprint(key) != cfg("key_fingerprint"):
        raise RuntimeError(
            "The mounted encryption key is not the key set in the app (fingerprints %s vs %s). "
            "Start the run again from the app." % (mcs_crypto.fingerprint(key), cfg("key_fingerprint")))
    _KEY["value"] = key
    return key


def local_model_dir():
    return cfg("local_model_dir", "/tmp/mcs_model")


def mark(msg):
    line = "[%7.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    try:
        with open(os.path.join(work_dir(), "debug.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------ backends

def backend():
    return cfg("backend", "tpu")


def fixed_canvas():
    return backend() == "tpu"


def get_device(rank):
    if backend() == "tpu":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    if backend() == "gpu":
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    return torch.device("cpu")


def sync(wait=True):
    """Flushes queued device work (XLA graphs) / waits for CUDA kernels."""
    if backend() == "tpu":
        import torch_xla
        torch_xla.sync(wait=wait)
    elif backend() == "gpu" and wait:
        torch.cuda.synchronize()


def mark_step():
    if backend() == "tpu":
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def mem_info(device):
    try:
        if backend() == "tpu":
            import torch_xla.core.xla_model as xm
            info = xm.get_memory_info(device)
            return "%.1f/%.1f GB" % (info["bytes_used"] / 1e9, info["bytes_limit"] / 1e9)
        if backend() == "gpu":
            return "%.1f/%.1f GB" % (torch.cuda.memory_allocated(device) / 1e9,
                                     torch.cuda.get_device_properties(device).total_memory / 1e9)
    except Exception:
        pass
    return "n/a"


def precision_plans():
    """Ordered {component: dtype} plans. A worker starts with the first and moves to the next
    if a page comes out blank (float16 overflow) or a plan fails outright.

    GPU (T4/P100 have no native bfloat16):
      1. transformer bfloat16 (emulated, but float32's range), the rest float32: ~7.4 GB.
         The Anima transformer overflows float16 on a T4 (NaN pages, seen on Kaggle), so
         bfloat16 goes first.
      2. transformer float16, the rest float32: faster where it doesn't overflow.
      3. everything float32: ~11.5 GB, tight on a 15 GB T4 even with VAE tiling.
    TPU: bfloat16 throughout. CPU (tests): cpu_dtype throughout.
    """
    def plan(transformer, rest):
        return {c: (transformer if c == "transformer" else rest) for c in COMPONENTS}

    if backend() == "gpu":
        return [plan(torch.bfloat16, torch.float32), plan(torch.float16, torch.float32),
                plan(torch.float32, torch.float32)]
    dt = getattr(torch, cfg("cpu_dtype", "bfloat16")) if backend() == "cpu" else torch.bfloat16
    return [plan(dt, dt)]


def describe_plan(plan):
    t, rest = plan["transformer"], plan["text_encoder"]
    name = lambda d: str(d).replace("torch.", "")
    return name(t) if t == rest else "transformer %s, rest %s" % (name(t), name(rest))


def world_size():
    if backend() == "tpu":
        return 8
    if backend() == "gpu":
        return max(1, torch.cuda.device_count())
    return 1


# ------------------------------------------------------------------ model & LoRA files

def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or ""
    except Exception:
        return ""


def prepare_local_model():
    """Finds the mounted model dataset and makes it loadable without network access.

    modular_model_index.json names each component's source repo, so the dataset copy would
    still be fetched from Hugging Face. A rewritten index points every component at the
    mounted folder instead.
    """
    root_dir = input_root()
    if not os.path.exists(root_dir):
        return None
    for root, dirs, files in os.walk(root_dir):
        if "modular_model_index.json" in files:
            with open(os.path.join(root, "modular_model_index.json")) as f:
                idx = json.load(f)
            for spec in idx.values():
                if isinstance(spec, list) and len(spec) == 3 and isinstance(spec[2], dict) and "pretrained_model_name_or_path" in spec[2]:
                    spec[2]["pretrained_model_name_or_path"] = root
                    spec[2]["revision"] = None
            os.makedirs(local_model_dir(), exist_ok=True)
            with open(os.path.join(local_model_dir(), "modular_model_index.json"), "w") as f:
                json.dump(idx, f, indent=2)
            return local_model_dir()
        if "model_index.json" in files:
            return root
    return None


def find_local_model():
    """The model folder prepared by prepare_local_model() in the main process, if any."""
    if os.path.isfile(os.path.join(local_model_dir(), "modular_model_index.json")):
        return local_model_dir()
    if os.path.exists(input_root()):
        for root, dirs, files in os.walk(input_root()):
            if "model_index.json" in files:
                return root
    return None


def _model_repo():
    return cfg("model_repo", "circlestone-labs/Anima-Base-v1.0-Diffusers")


def _model_revision():
    return cfg("model_revision") or "main"


def _download_with_watchdog(repo, filename, size, revision):
    """hf_hub_download in a subprocess that is killed if it stalls (HF CDN hiccups)."""
    code = ("from huggingface_hub import hf_hub_download; "
            "p = hf_hub_download(%r, %r, revision=%r); print('OK', p)" % (repo, filename, revision))
    timeout = 300 if size < 50 * 1024 * 1024 else max(600, int(size / (500 * 1024)))
    cache = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    def cache_bytes():
        total = 0
        for d in (cache / "hub", cache / "xet"):
            if d.is_dir():
                for p in d.rglob("*"):
                    try:
                        total += p.stat().st_size if p.is_file() else 0
                    except OSError:
                        pass
        return total

    t0 = last_progress = time.time()
    prev = cache_bytes()
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while proc.poll() is None:
        time.sleep(5)
        cur = cache_bytes()
        if cur > prev:
            prev, last_progress = cur, time.time()
        elif time.time() - last_progress > 180:
            proc.kill()
            mark("download of %s stalled for 3 minutes, retrying" % filename)
            return False
        if time.time() - t0 > timeout:
            proc.kill()
            mark("download of %s timed out after %ds, retrying" % (filename, timeout))
            return False
    out, err = proc.communicate()
    if proc.returncode != 0 or b"OK" not in out:
        mark("download of %s failed: %s" % (filename, err.decode(errors="replace")[-300:]))
        return False
    return True


def warm_model_cache():
    """Pre-downloads the model from Hugging Face when no cached dataset is mounted."""
    repo, revision = _model_repo(), _model_revision()
    url = "https://huggingface.co/api/models/%s/revision/%s?blobs=true" % (repo, revision)
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode())
    skip = {"README.md", "LICENSE.md", ".gitattributes"}
    files = {s["rfilename"]: (s.get("size") or 0) for s in data.get("siblings", []) if s["rfilename"] not in skip}
    for name in sorted(files, key=lambda n: files[n]):  # small config files first
        for attempt in range(1, 5):
            mark("downloading %s (attempt %d/4)..." % (name, attempt))
            if _download_with_watchdog(repo, name, files[name], revision):
                break
            time.sleep(2 * attempt)


def resolve_lora_path(lora_path_str):
    if os.path.isfile(lora_path_str):
        return lora_path_str
    filename = os.path.basename(lora_path_str).split("?")[0]

    # 1. Mounted Kaggle datasets & files downloaded earlier in this session
    if os.path.exists(input_root()):
        for root, dirs, files in os.walk(input_root()):
            if filename in files:
                return os.path.join(root, filename)
    lora_dir = os.path.join(work_dir(), "loras")
    if os.path.isfile(os.path.join(lora_dir, filename)):
        return os.path.join(lora_dir, filename)

    # 2. Direct URL
    if lora_path_str.startswith(("http://", "https://")):
        os.makedirs(lora_dir, exist_ok=True)
        dest = os.path.join(lora_dir, filename)
        mark("Downloading LoRA from URL: %s" % lora_path_str)
        headers = {"User-Agent": "Mozilla/5.0"}
        if os.getenv("HF_TOKEN") and "huggingface.co" in lora_path_str.split("/")[2]:
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        req = urllib.request.Request(lora_path_str, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
        return dest

    # 3. Hugging Face path: org/repo/file.safetensors
    if lora_path_str.count("/") >= 2:
        from huggingface_hub import hf_hub_download
        parts = lora_path_str.split("/")
        return hf_hub_download(repo_id="/".join(parts[:2]), filename="/".join(parts[2:]))

    raise FileNotFoundError("LoRA %r is not in any mounted dataset and has no download source." % lora_path_str)


# ------------------------------------------------------------------ LoRA fusion

_KOHYA_TO_DIFFUSERS = {
    "self_attn_q_proj": "attn1.to_q",
    "self_attn_k_proj": "attn1.to_k",
    "self_attn_v_proj": "attn1.to_v",
    "self_attn_output_proj": "attn1.to_out.0",
    "cross_attn_q_proj": "attn2.to_q",
    "cross_attn_k_proj": "attn2.to_k",
    "cross_attn_v_proj": "attn2.to_v",
    "cross_attn_output_proj": "attn2.to_out.0",
    "mlp_layer1": "ff.net.0.proj",
    "mlp_layer2": "ff.net.2",
}

_TURBO_MAP = {
    "self_attn.q_proj": "attn1.to_q",
    "self_attn.k_proj": "attn1.to_k",
    "self_attn.v_proj": "attn1.to_v",
    "self_attn.output_proj": "attn1.to_out.0",
    "cross_attn.q_proj": "attn2.to_q",
    "cross_attn.k_proj": "attn2.to_k",
    "cross_attn.v_proj": "attn2.to_v",
    "cross_attn.output_proj": "attn2.to_out.0",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
}


def lora_factors(sd):
    """{transformer weight name: (A, B, alpha)} for a Kohya or diffusion_model-style LoRA."""
    weights_A, weights_B, alphas = {}, {}, {}
    if any(k.startswith("lora_unet_blocks_") for k in sd.keys()):
        for key, value in sd.items():
            if not key.startswith("lora_unet_blocks_"):
                continue
            parts = key.split(".")
            tokens = parts[0][len("lora_unet_blocks_"):].split("_")
            diffusers_submod = _KOHYA_TO_DIFFUSERS.get("_".join(tokens[1:]))
            if diffusers_submod is None:
                continue
            target_name = "transformer_blocks.%s.%s.weight" % (tokens[0], diffusers_submod)
            rest = parts[1:]
            if rest == ["alpha"]:
                alphas[target_name] = value.item() if hasattr(value, "item") else float(value)
            elif len(rest) == 2 and rest[1] == "weight":
                if rest[0] == "lora_down":
                    weights_A[target_name] = value
                elif rest[0] == "lora_up":
                    weights_B[target_name] = value
    else:
        for k, v in sd.items():
            m = re.match(r"diffusion_model\.blocks\.(\d+)\.(.*?)\.(lora_[AB]\.weight|alpha)", k)
            if not m:
                continue
            block_idx, submod, param_type = m.groups()
            diff_sub = _TURBO_MAP.get(submod)
            if not diff_sub:
                continue
            target_name = "transformer_blocks.%s.%s.weight" % (block_idx, diff_sub)
            if param_type == "alpha":
                alphas[target_name] = v.item() if hasattr(v, "item") else float(v)
            elif "lora_A" in param_type:
                weights_A[target_name] = v
            else:
                weights_B[target_name] = v

    return {name: (A, weights_B[name], alphas.get(name, float(A.shape[0])))
            for name, A in weights_A.items() if name in weights_B}


def add_lora(lora_path, scale, transformer, tag=""):
    """Adds a LoRA's weight deltas straight into the transformer weights (no PEFT overhead).

    Called while the model is still on the CPU: thousands of small in-place adds are near
    instant there, whereas on TPU each one is a separate XLA graph (~5 minutes in total).
    Each full-size delta (B @ A) is built and applied one at a time: all of them at once
    would be ~7.5 GB of float32 per process, enough to exhaust a GPU machine's RAM.
    """
    from safetensors.torch import load_file
    file_path = resolve_lora_path(str(lora_path))
    factors = lora_factors(load_file(file_path))
    params = dict(transformer.named_parameters())
    fused = 0
    with torch.no_grad():
        for name, (A, B, alpha) in factors.items():
            param = params.get(name)
            if param is None:
                continue
            A, B = A.float(), B.float()
            delta = (B @ A) * ((alpha / A.shape[0]) * scale)
            if param.shape == delta.shape:
                param.add_(delta.to(device=param.device, dtype=param.dtype))
                fused += 1
            del delta
    mark("%s Fused LoRA %s (scale %.2f): %d matrices" % (tag, os.path.basename(file_path), scale, fused))
    return fused


# ------------------------------------------------------------------ pipeline

def install_reference_hook(pipe):
    """Feeds the encoded B&W page to the transformer as an extra frame, so the model colors
    the existing lineart instead of drawing a new image."""
    orig_forward = pipe.transformer.forward

    def reference_forward(hidden_states, timestep, encoder_hidden_states=None, **kwargs):
        ref_latent = getattr(pipe.transformer, "_ref_latent", None)
        if ref_latent is None:
            return orig_forward(hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
        is_4d = hidden_states.ndim == 4
        if is_4d:
            hidden_states = hidden_states.unsqueeze(2)
        out = orig_forward(hidden_states=torch.cat([hidden_states, ref_latent], dim=2), timestep=timestep,
                           encoder_hidden_states=encoder_hidden_states, **kwargs)
        if hasattr(out, "sample"):
            sample = out.sample[:, :, :1, :, :]
            out.sample = sample.squeeze(2) if is_4d else sample
            return out
        sample = out[0][:, :, :1, :, :]
        return (sample.squeeze(2) if is_4d else sample,)

    pipe.transformer.forward = reference_forward


def install_step_sync(pipe):
    """Cuts the XLA graph after every denoising step (no-op on GPU/CPU)."""
    sched = getattr(pipe, "scheduler", None)
    if sched is None or not hasattr(sched, "step"):
        return
    orig_step = sched.step

    def step(*args, **kwargs):
        out = orig_step(*args, **kwargs)
        mark_step()
        return out

    sched.step = step


def _load_components(pipe, plan, max_attempts=3):
    groups = {}
    for name, dt in plan.items():
        groups.setdefault(dt, []).append(name)
    for _ in range(max_attempts):
        for dt, names in groups.items():
            missing = [n for n in names if getattr(pipe, n, None) is None]
            if missing:
                pipe.load_components(names=missing, dtype=dt)
        # Loading by name skips components without weights, so ask for the scheduler explicitly.
        if getattr(pipe, "scheduler", None) is None:
            pipe.load_components(names=["scheduler"])
        if all(getattr(pipe, c, None) is not None for c in COMPONENTS + ["scheduler"]):
            return
        time.sleep(10)
    raise RuntimeError("Failed to load components %r" % [c for c in COMPONENTS + ["scheduler"] if getattr(pipe, c, None) is None])


def configure_guidance(pipe):
    """The Anima pipeline takes its CFG scale from its `guider` component (default 4.0), not
    from a call argument. At 1.0 the guider skips the unconditional pass, which halves the
    work per step and is what the Turbo LoRA is distilled for."""
    from diffusers.guiders import ClassifierFreeGuidance
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=float(cfg("guidance_scale", 1.0))))


def load_pipeline(plan, tag):
    from diffusers import ModularPipeline
    local = find_local_model()
    if local:
        try:
            pipe = ModularPipeline.from_pretrained(local)
            _load_components(pipe, plan)
            mark("%s Loaded model from mounted dataset (%s)" % (tag, describe_plan(plan)))
            return pipe
        except Exception as e:
            mark("%s Mounted model failed to load (%r); falling back to %s" % (tag, e, _model_repo()))
    pipe = ModularPipeline.from_pretrained(_model_repo(), revision=_model_revision())
    _load_components(pipe, plan)
    mark("%s Loaded model from Hugging Face (%s)" % (tag, describe_plan(plan)))
    return pipe


def build_pipeline(device, plan, tag):
    pipe = load_pipeline(plan, tag)
    configure_guidance(pipe)
    if backend() == "gpu":
        # Decode/encode in overlapping tiles: the full-page VAE pass is the memory peak on a T4.
        pipe.vae.enable_tiling()
    for lora in cfg("loras", []):
        add_lora(lora["path"], float(lora.get("scale", 1.0)), pipe.transformer, tag)
    pipe = pipe.to(device)
    sync(wait=True)
    install_reference_hook(pipe)
    install_step_sync(pipe)
    return pipe


# ------------------------------------------------------------------ pages

def canvas_size(w, h, max_dim):
    """Generation size for a page of w x h.

    TPU: one fixed portrait canvas for every page (any new shape triggers a slow XLA compile).
    GPU/CPU: the page's own aspect ratio at the same pixel budget, sides multiples of 32.
    """
    canvas_h = max(32, int(round(max_dim / 32.0)) * 32)
    canvas_w = max(32, int(round(max_dim * (1024.0 / 1440.0) / 32.0)) * 32)
    if fixed_canvas():
        return canvas_w, canvas_h
    s = math.sqrt(canvas_w * canvas_h / float(w * h))
    return max(32, int(w * s) // 32 * 32), max(32, int(h * s) // 32 * 32)


def prepare_canvas(img, max_dim):
    """Returns (canvas, crop_box): the page scaled (and on TPU letterboxed) for generation."""
    orig_w, orig_h = img.size
    canvas_w, canvas_h = canvas_size(orig_w, orig_h, max_dim)
    if not fixed_canvas():
        return img.resize((canvas_w, canvas_h), Image.LANCZOS), (0, 0, canvas_w, canvas_h)
    scale = min(canvas_w / orig_w, canvas_h / orig_h)
    scaled_w = max(1, min(canvas_w, int(round(orig_w * scale))))
    scaled_h = max(1, min(canvas_h, int(round(orig_h * scale))))
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    pad_x, pad_y = (canvas_w - scaled_w) // 2, (canvas_h - scaled_h) // 2
    canvas.paste(img.resize((scaled_w, scaled_h), Image.LANCZOS), (pad_x, pad_y))
    return canvas, (pad_x, pad_y, pad_x + scaled_w, pad_y + scaled_h)


def encode_reference(pipe, canvas, device):
    arr = np.array(canvas).astype(np.float32) / 127.5 - 1.0
    ref = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device=device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        enc = pipe.vae.encode(ref)
        if hasattr(enc, "latent_dist"):
            latent = enc.latent_dist.sample()
        elif isinstance(enc, tuple):
            latent = enc[0]
        else:
            latent = getattr(enc, "latents", enc)
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        vcfg = pipe.vae.config
        if getattr(vcfg, "latents_mean", None) is not None:
            mean = torch.tensor(vcfg.latents_mean, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(vcfg.latents_std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
            latent = (latent - mean) / std
        elif getattr(vcfg, "scaling_factor", None) is not None:
            latent = latent * vcfg.scaling_factor
    # joined with the transformer's own latents in the reference hook
    return latent.to(pipe.transformer.dtype)


def looks_broken(img):
    """float16 overflow shows up as a flat (usually black) image."""
    return float(np.asarray(img.convert("L"), dtype=np.float32).std()) < 2.0


def colorize_page(pipe, path, device):
    orig = Image.open(path).convert("RGB")
    canvas, crop_box = prepare_canvas(orig, int(cfg("max_dim", 1440)))
    w, h = canvas.size
    pipe.transformer._ref_latent = encode_reference(pipe, canvas, device)
    mark_step()
    kwargs = {
        "prompt": [cfg("prompt", "")],
        "num_inference_steps": int(cfg("num_steps", 8)),
        "height": h,
        "width": w,
    }
    if cfg("negative_prompt") and float(cfg("guidance_scale", 1.0)) > 1.0:
        kwargs["negative_prompt"] = [cfg("negative_prompt")]
    try:
        result = pipe(**kwargs)
        mark_step()
        out = result.images[0].crop(crop_box)
    finally:
        pipe.transformer._ref_latent = None
    return out.resize(orig.size, Image.LANCZOS) if out.size != orig.size else out


def release_memory():
    """Frees a pipeline the caller has already dropped all references to. The reference hook's
    closure makes a cycle (pipe -> transformer.forward -> pipe), so only a full gc frees it."""
    gc.collect()
    if backend() == "gpu":
        torch.cuda.empty_cache()


def mp_worker(rank, image_paths, n_workers):
    tag = "[%s %d/%d]" % ("Core" if backend() == "tpu" else backend().upper(), rank, n_workers)
    device = get_device(rank)
    plans = precision_plans()
    plan_idx = 0
    mark("%s Initializing pipeline on %s..." % (tag, device))
    pipe = build_pipeline(device, plans[plan_idx], tag)

    out_dir = os.path.join(pages_dir(), "generated")
    os.makedirs(out_dir, exist_ok=True)
    my_paths = image_paths[rank::n_workers]
    mark("%s Ready! Assigned %d images | memory: %s" % (tag, len(my_paths), mem_info(device)))

    t_start = time.time()
    for i, path in enumerate(my_paths):
        name = os.path.basename(path)
        out_path = os.path.join(out_dir, "edit_%s.png" % os.path.splitext(name)[0])
        if os.path.exists(out_path):
            continue
        t1 = time.time()
        mark("%s [%d/%d] %s -> generating..." % (tag, i + 1, len(my_paths), name))
        while True:
            try:
                colorized = colorize_page(pipe, path, device)
                problem = "a blank page (overflow)" if looks_broken(colorized) else None
            except Exception as e:
                traceback.print_exc()
                colorized, problem = None, repr(e)
            if problem is None:
                colorized.save(out_path)
                mark("%s [%d/%d] DONE %s (%ds, %s, mem %s)" % (tag, i + 1, len(my_paths), name, time.time() - t1,
                                                               describe_plan(plans[plan_idx]), mem_info(device)))
                break
            if plan_idx + 1 >= len(plans):
                mark("%s [%d/%d] FAILED %s: %s" % (tag, i + 1, len(my_paths), name, problem))
                try:
                    sync(wait=False)
                except Exception:
                    pass
                break
            # Switch precision for this page and every later one.
            plan_idx += 1
            mark("%s %s gave %s; switching to %s" % (tag, describe_plan(plans[plan_idx - 1]), problem,
                                                     describe_plan(plans[plan_idx])))
            pipe = colorized = None  # drop every reference first, then collect
            release_memory()
            try:
                pipe = build_pipeline(device, plans[plan_idx], tag)
            except Exception as e:
                traceback.print_exc()
                mark("%s could not build %s: %r" % (tag, describe_plan(plans[plan_idx]), e))
                if plan_idx + 1 >= len(plans):
                    mark("%s no precision left to try; stopping this worker" % tag)
                    return
        gc.collect()
    mark("%s ALL %d IMAGES FINISHED in %.0fs" % (tag, len(my_paths), time.time() - t_start))


# ------------------------------------------------------------------ orchestration

def collect_scans():
    unzip_dir = os.path.join(pages_dir(), "input_pages")
    os.makedirs(unzip_dir, exist_ok=True)
    if os.path.exists(input_root()):
        for root, dirs, files in os.walk(input_root()):
            for name in files:
                if not name.endswith(".mcsenc"):
                    continue
                import mcs_crypto
                if _KEY["value"] is None:
                    raise RuntimeError("Found encrypted scans (%s) but encryption is off for this run." % name)
                plain = os.path.join(pages_dir(), "scans.zip")
                mcs_crypto.decrypt_file(os.path.join(root, name), plain, _KEY["value"])
                with zipfile.ZipFile(plain) as z:
                    z.extractall(unzip_dir)
                os.remove(plain)
                print("[Decrypt] Decrypted and extracted '%s'" % name, flush=True)
            for zf in files:
                if zf.lower().endswith(".zip") and "model" not in zf.lower():
                    try:
                        with zipfile.ZipFile(os.path.join(root, zf)) as z:
                            z.extractall(unzip_dir)
                        print("[Auto-Unzip] Extracted '%s'" % zf, flush=True)
                    except Exception as ze:
                        print("[Auto-Unzip Warning] %s: %s" % (zf, ze), flush=True)

    paths = []
    for root, dirs, files in os.walk(unzip_dir):
        paths += [os.path.join(root, f) for f in files if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")]
    if not paths and os.path.exists(input_root()):
        for root, dirs, files in os.walk(input_root()):
            if "model" in root.lower() or "lora" in root.lower():
                continue
            paths += [os.path.join(root, f) for f in files if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")]
    if not paths:
        for root, dirs, files in os.walk(input_root()):
            print("  %s -> %s" % (root, files), flush=True)
        raise FileNotFoundError("No manga scans found in %s" % input_root())
    return sorted(paths)


def split_spreads(paths):
    """TPU only: splits double-page spreads into two portrait halves (one canvas shape)."""
    if not fixed_canvas():
        return list(paths), {}
    split_dir = os.path.join(pages_dir(), "split_pages")
    os.makedirs(split_dir, exist_ok=True)
    queue, spreads = [], {}
    for p in paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
                if w <= h * 1.15:
                    queue.append(p)
                    continue
                stem = os.path.splitext(os.path.basename(p))[0]
                halves = []
                for side, box in (("left", (0, 0, w // 2, h)), ("right", (w // 2, 0, w, h))):
                    half = os.path.join(split_dir, "%s__spread_%s.png" % (stem, side))
                    img.crop(box).save(half)
                    halves.append(half)
                queue += halves
                spreads[stem] = (w, h)
                print("[Auto-Split] '%s' (%dx%d) split into two halves." % (os.path.basename(p), w, h), flush=True)
        except Exception as e:
            print("Error inspecting '%s': %s" % (p, e), flush=True)
            queue.append(p)
    return queue, spreads


def stitch_spreads(spreads):
    out_dir = os.path.join(pages_dir(), "generated")
    for stem, (orig_w, orig_h) in spreads.items():
        left = os.path.join(out_dir, "edit_%s__spread_left.png" % stem)
        right = os.path.join(out_dir, "edit_%s__spread_right.png" % stem)
        if not (os.path.exists(left) and os.path.exists(right)):
            print("[Auto-Stitch Warning] Missing halves for '%s'" % stem, flush=True)
            continue
        l_img, r_img = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
        h = max(l_img.height, r_img.height)
        l_img = l_img.resize((int(l_img.width * h / l_img.height), h), Image.LANCZOS)
        r_img = r_img.resize((int(r_img.width * h / r_img.height), h), Image.LANCZOS)
        stitched = Image.new("RGB", (l_img.width + r_img.width, h))
        stitched.paste(l_img, (0, 0))
        stitched.paste(r_img, (l_img.width, 0))
        if stitched.size != (orig_w, orig_h):
            stitched = stitched.resize((orig_w, orig_h), Image.LANCZOS)
        stitched.save(os.path.join(out_dir, "edit_%s.png" % stem))
        os.remove(left)
        os.remove(right)
        print("[Auto-Stitch] Stitched 'edit_%s.png' (%dx%d)" % (stem, orig_w, orig_h), flush=True)


def package_outputs():
    """Zips the colorized pages into one archive in /kaggle/working (encrypted when encryption
    is on) and removes every scratch file. Returns the archive's file name."""
    work, pages = work_dir(), pages_dir()
    out_dir = os.path.join(pages, "generated")
    archive = "generated.zip"
    if os.path.isdir(out_dir):
        plain = os.path.join(pages, "generated.zip")
        with zipfile.ZipFile(plain, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(os.listdir(out_dir)):
                zf.write(os.path.join(out_dir, f), arcname="generated/" + f)
        shutil.rmtree(out_dir, ignore_errors=True)
        if encrypted():
            import mcs_crypto
            archive = "generated" + mcs_crypto.EXTENSION
            mcs_crypto.encrypt_file(plain, os.path.join(work, archive), _KEY["value"])
            os.remove(plain)
        elif plain != os.path.join(work, archive):
            shutil.move(plain, os.path.join(work, archive))
    for base in {work, pages}:
        for name in ("split_pages", "input_pages", "loras", "colorize_worker.py", "mcs_crypto.py"):
            p = os.path.join(base, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.remove(p)
    return archive


def spawn_workers(queue):
    n = world_size()
    if backend() != "tpu":
        n = max(1, min(n, len(queue)))  # an idle GPU process would still load the whole model
    if backend() == "tpu":
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(mp_worker, args=(queue, n), nprocs=None)
    elif n > 1:
        import torch.multiprocessing as mp
        mp.spawn(mp_worker, args=(queue, n), nprocs=n, join=True)
    else:
        mp_worker(0, queue, 1)
    return n


def main():
    hf_token = os.getenv("HF_TOKEN") or secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.makedirs(work_dir(), exist_ok=True)
    os.makedirs(pages_dir(), exist_ok=True)
    mark("Backend: %s | model: %s | %d LoRA(s)%s" % (backend(), _model_repo(), len(cfg("loras", [])),
                                                     " | encrypted" if encrypted() else ""))
    if encrypted():
        load_key()  # fail in seconds, before any model download, if the secret is missing or wrong
        mark("Encryption key verified (fingerprint %s)." % cfg("key_fingerprint"))

    if prepare_local_model():
        mark("Model found in mounted dataset; no download needed.")
    else:
        mark("No cached model mounted; downloading %s from Hugging Face..." % _model_repo())
        warm_model_cache()

    queue, spreads = split_spreads(collect_scans())
    mark("Work queue: %d pages (%d spreads split)" % (len(queue), len(spreads)))

    # Resolve every LoRA once, so the worker processes don't all download the same file.
    for lora in cfg("loras", []):
        print("[LoRA] %s -> %s" % (lora["path"], resolve_lora_path(lora["path"])), flush=True)

    t_start = time.time()
    try:
        n = spawn_workers(queue)
        mark("ALL %d WORKERS COMPLETED in %.0fs" % (n, time.time() - t_start))
    except Exception as e:
        mark("[Warning] Worker run ended with: %r" % e)
    stitch_spreads(spreads)
    archive = package_outputs()
    mark("Outputs packaged into %s" % archive)
