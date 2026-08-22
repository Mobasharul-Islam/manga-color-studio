import sys
import socket
import webbrowser
import threading
import time
from pathlib import Path

# Ensure root directory is on python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn


def find_free_port(start_port=7860, max_tries=10):
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', p))
                return p
            except OSError:
                continue
    return start_port


def open_browser(port):
    time.sleep(1.2)
    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f"  Manga Color Studio (8-Core TPU v5-8 Accelerator)")
    print(f"  Web Dashboard running at: {url}")
    print(f"=======================================================\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print("Could not auto-open browser:", e)


if __name__ == "__main__":
    port = find_free_port(7860)
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    uvicorn.run("app.server:app", host="127.0.0.1", port=port, reload=False)
