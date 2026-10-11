"""
Step 3: live voice controller = the bridge between the microphone and the game.

Needs (in the same folder):  features.py  and  model.joblib (from train_model.py)

How the game uses it (3 lines):

    from voice_controller import VoiceController
    voice = VoiceController()          # VoiceController(device=1) to pick a mic
    voice.start()
    ...
    # once per game frame:
    cmd = voice.get_command()          # "up" / "down" / "left" / "right" or None
    if cmd:
        direction = cmd                # ball keeps moving; voice only changes direction
    ...
    voice.stop()                       # when the game closes

Test it alone first:   python voice_controller.py        (add --device 1 if needed)

How it works: the microphone is listened to continuously. When the sound level
jumps above `onset_rms`, a word has started. We wait 0.7 s, then cut a 1-second
clip (0.3 s before the start + 0.7 s after) - the same layout as the training
recordings - and classify it. Silence/noise classes and low-confidence guesses
are ignored, so the game only ever receives the four commands.
"""

import argparse
import queue
import threading
import time
from math import gcd

import joblib
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from features import SAMPLE_RATE, extract_features

COMMANDS = ("up", "down", "left", "right")

BLOCK_SECONDS = 0.02   # microphone is read in 20 ms blocks
PRE_SECONDS = 0.3      # audio kept from BEFORE the detected word start
POST_SECONDS = 0.5     # audio kept AFTER the detected word start


class VoiceController:
    def __init__(self, model_path="model.joblib", device=None,
                 onset_rms=800.0, min_confidence=0.6,
                 verbose=False, on_command=None):
        """
        onset_rms      sound level (int16 units) that counts as "a word started".
                       Raise it if noise triggers it, lower it if words are missed.
        min_confidence needs a model trained with SVC(probability=True);
                       otherwise it is ignored.
        verbose        print every prediction (useful while tuning).
        on_command     optional function called with the command string.
        """
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.device = device
        self.onset_rms = onset_rms
        self.min_confidence = min_confidence
        self.verbose = verbose
        self.on_command = on_command

        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._command = None
        self._stream = None
        self._worker = None
        self._running = False
        self._rate = SAMPLE_RATE
        extract_features(np.zeros(SAMPLE_RATE, dtype=np.int16))   # warm-up

    # ---------- public API ----------
    def start(self):
        if self._running:
            return
        self._stream = self._open_stream()
        self._running = True
        self._worker = threading.Thread(target=self._process, daemon=True)
        self._worker.start()
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None

    def get_command(self):
        """Return the newest unread command, or None. Never blocks."""
        with self._lock:
            cmd, self._command = self._command, None
        return cmd

    # ---------- microphone ----------
    def _open_stream(self):
        rates = [SAMPLE_RATE]
        native = int(sd.query_devices(self.device, "input")["default_samplerate"])
        if native != SAMPLE_RATE:
            rates.append(native)  # fallback if the mic refuses 16 kHz
        last_error = None
        for rate in rates:
            try:
                stream = sd.InputStream(
                    samplerate=rate, channels=1, dtype="int16",
                    blocksize=int(rate * BLOCK_SECONDS),
                    device=self.device, callback=self._on_audio,
                    latency="low",
                )
                self._rate = rate
                return stream
            except sd.PortAudioError as e:
                last_error = e
        raise RuntimeError(f"Could not open the microphone: {last_error}")

    def _on_audio(self, indata, frames, time_info, status):
        self._queue.put(indata[:, 0].copy())

    # ---------- word detection ----------
    def _process(self):
        rate = self._rate
        pre = int(PRE_SECONDS * rate)
        post = int(POST_SECONDS * rate)
        history = np.zeros(0, dtype=np.int16)
        collecting = False
        onset_idx = 0
        armed = True   # re-armed only after a short quiet moment
        quiet = 0

        while self._running:
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            level = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
            history = np.concatenate([history, block])

            if collecting:
                if len(history) - onset_idx >= post:
                    clip = self._cut_clip(history, onset_idx, pre, post)
                    self._classify(clip, rate)
                    collecting = False
                    armed = False
                    quiet = 0
                    history = history[-rate:]
            else:
                if len(history) > rate:
                    history = history[-rate:]
                if not armed:
                    if level < self.onset_rms * 0.5:
                        quiet += 1
                        if quiet >= 5:   # ~100 ms of calm
                            armed = True
                    else:
                        quiet = 0
                elif level > self.onset_rms:
                    collecting = True
                    onset_idx = len(history) - len(block)

    @staticmethod
    def _cut_clip(history, onset_idx, pre, post):
        start = onset_idx - pre
        clip = history[max(start, 0): onset_idx + post]
        if start < 0:  # program just started: pad the front with silence
            clip = np.concatenate([np.zeros(-start, dtype=history.dtype), clip])
        return clip

    # ---------- classification ----------
    def _classify(self, clip, rate):
        if rate != SAMPLE_RATE:
            g = gcd(SAMPLE_RATE, rate)
            clip = resample_poly(clip.astype(np.float32), SAMPLE_RATE // g, rate // g)
            clip = np.clip(clip, -32768, 32767).astype(np.int16)

        x = extract_features(clip).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            i = int(np.argmax(proba))
            label, conf = str(self.model.classes_[i]), float(proba[i])
        else:
            label, conf = str(self.model.predict(x)[0]), 1.0

        if self.verbose:
            print(f"[voice] heard: {label} (confidence {conf:.2f})")
        self.last_confidence = conf
        if label in COMMANDS and conf >= self.min_confidence:
            with self._lock:
                self._command = label
            if self.on_command:
                self.on_command(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the voice controller alone.")
    parser.add_argument("--device", type=int, default=None, help="microphone number")
    args = parser.parse_args()

    voice = VoiceController(device=args.device, verbose=True)
    voice.start()
    print("Listening... say up / down / left / right. Ctrl+C to stop.")
    try:
        while True:
            cmd = voice.get_command()
            if cmd:
                print(">>> command:", cmd)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
