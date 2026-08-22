import builtins
import os
import sys

# Monkey-patch builtins.open to default to utf-8 when encoding is not specified
orig_open = builtins.open

def utf8_open(*args, **kwargs):
    if len(args) > 1 and 'b' in args[1]:
        return orig_open(*args, **kwargs)
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
        kwargs['errors'] = 'replace'
    return orig_open(*args, **kwargs)

builtins.open = utf8_open

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

kernel = "mobasharulislam/tpu-image-generation-anima-base-1-0-2b"
dest = sys.argv[1] if len(sys.argv) > 1 else "output_anima/out_v21"
os.makedirs(dest, exist_ok=True)

print("Downloading kernel output to:", dest)
api.kernels_output(kernel, path=dest, force=True)
print("Downloaded files:", os.listdir(dest))

# Auto-extract any downloaded zip archives into dest
import zipfile
for fname in os.listdir(dest):
    if fname.endswith(".zip"):
        zpath = os.path.join(dest, fname)
        print(f"Extracting {fname} into {dest}...")
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(dest)
            print(f"Extracted {fname} successfully.")
        except Exception as ze:
            print(f"Failed to extract {fname}: {ze}")

# Also read the log file if present
log_files = [f for f in os.listdir(dest) if f.endswith('.log')]
for lf in log_files:
    print(f"\n=== Log file {lf} saved successfully ===")
    with orig_open(os.path.join(dest, lf), 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        print("Total log chars:", len(content))
        lines = content.splitlines()
        print("Last 30 lines:")
        for l in lines[-30:]:
            try:
                print(l)
            except Exception:
                print(l.encode('ascii', 'replace').decode('ascii'))

