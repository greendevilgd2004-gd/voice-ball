"""
Feature extraction shared by train_model.py and the live voice controller.

A 1-second, 16 kHz signal -> one fixed-length feature vector:
  13 MFCCs + 13 delta-MFCCs, each summarised by mean and std over time
  = 52 numbers.
"""

import librosa
import numpy as np

SAMPLE_RATE = 16000
N_SAMPLES = SAMPLE_RATE          # exactly 1 second
N_MFCC = 13
FRAME_LENGTH = 400               # 25 ms window
HOP_LENGTH = 160                 # 10 ms step
PRE_SECONDS = 0.3     # باید با voice_controller.py یکی باشد
POST_SECONDS = 0.7    # باید با voice_controller.py یکی باشد


def to_float(signal):
    """int16 wav data -> float32 in [-1, 1]."""
    signal = np.asarray(signal)
    if signal.dtype == np.int16:
        return signal.astype(np.float32) / 32768.0
    return signal.astype(np.float32)


def extract_features(signal):
    signal = to_float(signal)
    signal = librosa.util.fix_length(signal, size=N_SAMPLES)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)

    start = int(PRE_SECONDS * SAMPLE_RATE / HOP_LENGTH)           # first frame of the word region
    parts = np.array_split(np.arange(start, mfcc.shape[1]), 3)    # 3 time segments
    return np.concatenate(
        [mfcc[:, p].mean(axis=1) for p in parts]
        + [delta[:, p].mean(axis=1) for p in parts]
        + [mfcc.std(axis=1), delta.std(axis=1)]
    )



def align_clip(signal, onset_rms=800.0):
    """Cut a recorded clip the same way the live controller does."""
    signal = np.asarray(signal)
    block = int(0.02 * SAMPLE_RATE)
    pre = int(PRE_SECONDS * SAMPLE_RATE)
    post = int(POST_SECONDS * SAMPLE_RATE)
    x = signal.astype(np.float64)
    onset = None
    for i in range(0, len(x) - block + 1, block):
        if np.sqrt(np.mean(x[i:i + block] ** 2)) > onset_rms:
            onset = i
            break
    if onset is None:                      # nothing loud: keep the start of the clip
        return signal[: pre + post]
    start = onset - pre
    clip = signal[max(start, 0): onset + post]
    if start < 0:
        clip = np.concatenate([np.zeros(-start, dtype=signal.dtype), clip])
    return clip
