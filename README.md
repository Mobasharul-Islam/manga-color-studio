# Manga Color Studio 🎨 (Kaggle TPU & GPU Pipeline)

High-performance manga chapter colorization powered by **Anima Base 1.0 (2B)** running across **8 TPU Cores (TPU v5-8)** in parallel via PyTorch XLA (PJRT).

---

## ✨ Key Features

- **🚀 TPU or GPU**: Run on Kaggle's 8-core TPU v5e-8 (fastest for long chapters) or on GPU T4 x2 / P100 (usually starts within minutes when the TPU queue is long). The app shows your remaining weekly hours for each.
- **⚡ 1-Click Launch**: Double-click `run.bat` — all dependencies and local web servers are configured automatically.
- **🔑 In-UI API Token Setup**: Drag & drop your `kaggle.json` or paste your Kaggle & Hugging Face tokens directly into the browser without editing config files.
- **📦 Instant Chapter Ingestion**: Drag & drop folder directories or `.zip` / `.cbz` archives with in-memory single-stream packaging.
- **🎛️ Model & LoRA Library**: Add models and LoRAs by link or upload, see what is cached on Kaggle, and delete what you no longer need. Each item lives in its own private Kaggle dataset, cached by a CPU-only notebook, so TPU runs boot without downloading. Live weight sliders and style presets (*Vibrant Shonen*, *Pastel Shoujo*, *Cinematic*, *Chisa Kotegawa*).
- **🖋️ Lineart & Speech Bubble Restore**: Multiplies the original scan's lines over the color and pastes detected speech bubbles back from the scan, so lines and lettering stay crisp. Runs locally with a live preview slider; no TPU time needed to tune it.
- **🔁 Re-run Selected Pages**: Tick bad pages in Grid View (or use *Re-run this page*) to colorize just those again with the current prompt and LoRAs. Replaced versions are kept in the chapter's `history/` folder.
- **🔬 Studio Comparison Viewer**: Interactive B&W vs. Color split slider, dual-stage side-by-side view, all-pages gallery, and keyboard navigation (`←`/`→` or `A`/`D`).

---

## 🏁 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/manga-color-studio.git
cd manga-color-studio
```

### 2. Launch
Double-click **`run.bat`** (or run `python run_app.py` in your terminal).
Your default web browser will automatically open to:
```
http://localhost:7860
```

### 3. Connect Kaggle Account
1. Click **`API Tokens`** in the top-right header (or the prompt when launching).
2. Drag & drop your **`kaggle.json`** (from [kaggle.com/settings](https://www.kaggle.com/settings) ➔ *Create New Token*) or enter your username & API key.
3. *(Optional)* Enter a free [Hugging Face Read Token](https://huggingface.co/settings/tokens) to download base Anima weights at high speed.

### 4. Model & LoRA Library
The **Model, LoRAs & Style** card lists every model and LoRA, whether it is cached on Kaggle, and lets you add or delete them.

- **Add model**: paste a Hugging Face repo of an Anima model in Diffusers format (e.g. `circlestone-labs/Anima-Base-v1.0-Diffusers`). It is checked on your PC first, then a small CPU-only Kaggle notebook (*Manga Color Library Worker*) downloads it and saves it as its own private dataset (`mcs-model-<name>`). No TPU quota or TPU queue is involved. Single-file checkpoints (e.g. most Civitai models) aren't supported yet.
- **Add LoRA**: by Hugging Face path, Hugging Face / Civitai / direct link, or upload a `.safetensors` from your PC. Each LoRA gets its own private dataset (`mcs-lora-<name>`), so adding or deleting one never re-uploads the others. LoRA weights you set with the sliders are remembered.
- **Delete** removes the item and permanently deletes its dataset (after a confirmation). Datasets created by older versions of the app are listed under *Older app datasets on Kaggle* and can be deleted there.
- TPU runs mount only the selected model and the enabled LoRAs. Anything not cached is downloaded from Hugging Face during the run.

**One-time Kaggle Secrets setup.** API keys are never written into a pushed notebook. The first time the library caches something from a link, the run stops and asks for a secret: open *Manga Color Library Worker* on kaggle.com, go to **Add-ons → Secrets** and attach:

| Secret | Notebook | Purpose |
|---|---|---|
| `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`) | Library Worker | Required to save datasets |
| `HF_TOKEN` | Library Worker, TPU notebook | Optional: gated repos / faster Hugging Face downloads |
| `CIVITAI_TOKEN` | Library Worker | Optional: mature Civitai LoRAs (or enter your Civitai key in the app, which then downloads on your PC) |

### 5. Choose Hardware
| Option | Starts | Notes |
|---|---|---|
| **TPU v5e-8** | The TPU queue can take hours on busy days | 8 cores in parallel; fastest on long chapters. ~4 min one-time compile per run. Double-page spreads are colorized as two halves. |
| **GPU T4 x2** | Usually within minutes | 2 GPUs in parallel. Double-page spreads are colorized whole. Runs in float16, switching to float32 automatically if a page comes out blank. |
| **GPU P100** | Usually within minutes | 1 GPU, roughly half the speed of T4 x2. |

TPU and GPU hours are separate weekly Kaggle quotas. Each option uses its own Kaggle notebook, so a run waiting in one queue never blocks another.

### 6. Colorize a Chapter
1. Click **Select Folder** (or drop a `.zip`/`.cbz` file).
2. Adjust positive prompts or select a style preset.
3. Pick the **Hardware** and click **Colorize Chapter**.
4. Monitor real-time logs and compare before/after results in the comparison studio!
5. Adjust **Lineart** / **Keep speech bubbles** and click **Apply to chapter**; re-run any page that came out wrong.

Each chapter is stored under `output_app/<chapter>/`: `input/` (original scans), `generated/` (raw TPU output), `final/` (post-processed pages) and `history/` (replaced versions).

---

## 🛠️ Requirements

- **OS**: Windows 10/11, macOS, or Linux.
- **Python**: Python 3.9+ (with `pip`).
- **Kaggle Account**: Free Kaggle account with phone verification (enables free 30 hours/week of TPU v5-8 accelerator access).

---

## 📜 License
MIT License.
