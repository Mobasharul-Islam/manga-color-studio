# Manga Color Studio 🎨 (Distributed 8-Core TPU v5-8 Pipeline)

High-performance manga chapter colorization powered by **Anima Base 1.0 (2B)** running across **8 TPU Cores (TPU v5-8)** in parallel via PyTorch XLA (PJRT).

---

## ✨ Key Features

- **🚀 Distributed 8-Core Parallelism**: Shards full manga chapters evenly across 8 hardware TPU cores, delivering complete 50+ page chapter colorization in minutes.
- **⚡ 1-Click Launch**: Double-click `run.bat` — all dependencies and local web servers are configured automatically.
- **🔑 In-UI API Token Setup**: Drag & drop your `kaggle.json` or paste your Kaggle & Hugging Face tokens directly into the browser without editing config files.
- **📦 Instant Chapter Ingestion**: Drag & drop folder directories or `.zip` / `.cbz` archives with in-memory single-stream packaging.
- **🎛️ Dynamic Multi-LoRA Stack**: Live sliders and toggle switches for custom character and style LoRAs, zero-wait cloud caching, and style steering presets (*Vibrant Shonen*, *Pastel Shoujo*, *Cinematic*, *Chisa Kotegawa*).
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

### 4. Colorize a Chapter
1. Click **Select Folder** (or drop a `.zip`/`.cbz` file).
2. Adjust positive prompts or select a style preset.
3. Click **Colorize Chapter on 8-Core TPU**.
4. Monitor real-time logs and compare before/after results in the comparison studio!

---

## 🛠️ Requirements

- **OS**: Windows 10/11, macOS, or Linux.
- **Python**: Python 3.9+ (with `pip`).
- **Kaggle Account**: Free Kaggle account with phone verification (enables free 30 hours/week of TPU v5-8 accelerator access).

---

## 📜 License
MIT License.
