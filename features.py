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

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
    ])
