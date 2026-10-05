"""
Step 1: record the voice-command dataset.

Install:  pip install sounddevice scipy numpy
Run:      python record_data.py

Each recording is 1 second, 16 kHz, mono, saved as:
    data/<label>/<label>_<number>.wav
"""

import os

import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000   # 16 kHz is standard for speech
DURATION = 1.0        # seconds per recording
DATA_DIR = "data"
TARGET = 50           # samples to record per label

# folder name -> what you should say
LABELS = {
    "up": "بالا",
    "down": "پایین",
    "left": "چپ",
    "right": "راست",
    "silence": "(چیزی نگو، فقط نویز محیط)",
}


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


def record_one():
    """Record DURATION seconds from the default microphone."""
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()  # block until the recording is finished
    return audio.squeeze()


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
        audio = record_one()
        idx = next_index(folder, label)
        last_path = os.path.join(folder, f"{label}_{idx:03d}.wav")
        wavfile.write(last_path, SAMPLE_RATE, audio)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
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
