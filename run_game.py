"""
One-step launcher:   python run_game.py

1) Trains the model if it is missing, or if you recorded samples newer than it.
2) Starts the voice server (voice_server.py).
3) Opens the game page in your browser as soon as the server is ready.
Ctrl+C stops everything.

Options:
  --device 1      microphone number
  --onset 1500    word-start sensitivity (higher = less sensitive)
  --game page.html   game file (default: the only .html file in this folder)
  --no-train      do not retrain even if there are newer recordings
"""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model.joblib"
DATA = HERE / "data"
READY_TEXT = "Voice server running"


def newest_sample_time():
    return max((p.stat().st_mtime for p in DATA.rglob("*.wav")), default=0)


def ensure_model(skip_training):
    newest = newest_sample_time()
    model_exists = MODEL.exists()
    stale = model_exists and newest > MODEL.stat().st_mtime

    if model_exists and not stale:
        print("Model is up to date.")
        return
    if model_exists and stale and skip_training:
        print("New recordings found, but --no-train was given: using the old model.")
        return
    if not model_exists and newest == 0:
        sys.exit("No model and no recordings found. Run record_data.py first.")

    print("Training the model..." if not model_exists
          else "New recordings found -> retraining the model...")
    result = subprocess.run([sys.executable, "train_model.py"], cwd=HERE)
    if result.returncode != 0:
        if model_exists:
            print("Training failed. Continuing with the previous model.")
        else:
            sys.exit("Training failed and there is no model to use.")


def find_game(arg):
    if arg:
        path = Path(arg)
        if not path.is_absolute():
            path = HERE / path
    else:
        pages = sorted(HERE.glob("*.html"))
        if len(pages) != 1:
            sys.exit("Put exactly one .html file in this folder, or use --game <file>.")
        path = pages[0]
    if not path.exists():
        sys.exit(f"Game page not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Start model, voice server and game.")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--onset", type=float, default=None)
    parser.add_argument("--game", default=None)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    game = find_game(args.game)
    ensure_model(args.no_train)

    cmd = [sys.executable, "-u", "voice_server.py"]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    if args.onset is not None:
        cmd += ["--onset", str(args.onset)]

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    server = subprocess.Popen(
        cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )

    opened = False
    try:
        for line in server.stdout:          # show the server's messages here
            print(line, end="")
            if not opened and READY_TEXT in line:
                opened = True
                print(">>> Opening the game:", game)
                if not webbrowser.open(game.as_uri()):
                    print("Could not open the browser automatically. Open this file yourself:", game)
        code = server.wait()
        if not opened:
            print(f"The voice server stopped before it was ready (exit code {code}). "
                  "Read the messages above.")
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
        print("Stopped.")


if __name__ == "__main__":
    main()
