"""
Step 1: record the voice-command dataset (version 2, with diagnostics).

Install:  pip install sounddevice scipy numpy
Run:      python record_data.py

Each recording is 1 second, 16 kHz, mono, saved as:
    data/<label>/<label>_<number>.wav
The data folder is created NEXT TO THIS SCRIPT (full path is printed),
no matter which folder the terminal is in.
"""

import math
import os

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000   # 16 kHz is standard for speech
DURATION = 1.0        # seconds per recording
TARGET = 50           # samples to record per label
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QUIET_PEAK = 1000     # int16 peak below this = probably no speech captured

# folder name -> what you should say
LABELS = {
    "up": "بالا",
    "down": "پایین",
    "left": "چپ",
    "right": "راست",
    "silence": "(چیزی نگو، فقط نویز محیط)",
}

INPUT_DEVICE = None   # None = system default microphone


def choose_device():
    """Let the user pick the microphone (Enter = default)."""
    global INPUT_DEVICE
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    print("\nInput devices:")
    for i, d in inputs:
        print(f"  [{i}] {d['name']}")
    default = sd.default.device[0]
    choice = input(f"Device number (Enter = default [{default}]): ").strip()
    if choice.isdigit() and int(choice) in [i for i, _ in inputs]:
        INPUT_DEVICE = int(choice)
    info = sd.query_devices(INPUT_DEVICE, "input")
    print("Using microphone:", info["name"])


def next_index(folder, label):
    """Next free file number, so re-running never overwrites old samples."""
    nums = []
    for name in os.listdir(folder):
        if name.startswith(label + "_") and name.endswith(".wav"):
            try:
                nums.append(int(name[len(label) + 1:-4]))
            except ValueError:
                pass
    return max(nums, default=-1) + 1


def count_samples(folder):
    return len([f for f in os.listdir(folder) if f.endswith(".wav")])


def _rec(rate):
    audio = sd.rec(
        int(DURATION * rate),
        samplerate=rate,
        channels=1,
        dtype="int16",
        device=INPUT_DEVICE,
    )
    sd.wait()  # block until the recording is finished
    return audio.squeeze()


def record_one():
    """Record DURATION seconds; fall back to the device's own rate if needed."""
    try:
        return _rec(SAMPLE_RATE)
    except sd.PortAudioError as e:
        info = sd.query_devices(INPUT_DEVICE, "input")
        native = int(info["default_samplerate"])
        print(f"16 kHz not supported here ({e}). Recording at {native} Hz and converting.")
        audio = _rec(native)
        g = math.gcd(SAMPLE_RATE, native)
        audio = resample_poly(audio.astype(np.float32), SAMPLE_RATE // g, native // g)
        return np.clip(audio, -32768, 32767).astype(np.int16)


def record_label(label):
    folder = os.path.join(DATA_DIR, label)
    os.makedirs(folder, exist_ok=True)
    last_path = None

    while True:
        count = count_samples(folder)
        if count >= TARGET:
            print(f"'{label}' is done ({count} samples).")
            return

        cmd = input(
            f"[{label}: {count}/{TARGET}] say '{LABELS[label]}' | "
            "Enter=record, d=delete last, q=back: "
        ).strip().lower()

        if cmd == "q":
            return
        if cmd == "d":
            if last_path and os.path.exists(last_path):
                os.remove(last_path)
                print("Deleted", last_path)
                last_path = None
            else:
                print("Nothing to delete.")
            continue

        print(">>> Speak now!")
        try:
            audio = record_one()
        except Exception as e:  # show the real problem instead of failing silently
            print("RECORDING FAILED:", e)
            continue

        peak = int(np.abs(audio.astype(np.int32)).max())
        idx = next_index(folder, label)
        path = os.path.join(folder, f"{label}_{idx:03d}.wav")
        wavfile.write(path, SAMPLE_RATE, audio)
        print(f"Saved: {path}  (loudness peak: {peak})")
        if label != "silence" and peak < QUIET_PEAK:
            print("WARNING: very quiet. Wrong microphone, too far, or muted? "
                  "Press d to delete this one.")
        last_path = path


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Dataset folder:", DATA_DIR)
    choose_device()
    while True:
        print("\nLabels:")
        for label, word in LABELS.items():
            folder = os.path.join(DATA_DIR, label)
            n = count_samples(folder) if os.path.isdir(folder) else 0
            print(f"  {label:8s} {word:30s} {n}/{TARGET}")
        choice = input("Which label to record? (name, or q to quit): ").strip()
        if choice == "q":
            break
        if choice in LABELS:
            record_label(choice)
        else:
            print("Unknown label.")


if __name__ == "__main__":
    main()
