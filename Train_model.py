"""
Step 2: train the voice-command classifier.

Install:  pip install librosa scikit-learn joblib matplotlib scipy numpy
Run:      python train_model.py

Reads   data/<label>/*.wav   (made by record_data.py)
Writes  model.joblib         (used later by the live controller)
        confusion_matrix.png (for the report)
"""

import os

import joblib
import matplotlib
import numpy as np
from scipy.io import wavfile
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import align_clip, extract_features

matplotlib.use("Agg")  # save figures to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = "data"
LABELS = ["up", "down", "left", "right", "silence"]


def load_dataset():
    X, y = [], []
    for label in LABELS:
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            print(f"WARNING: folder missing: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        for name in files:
            rate, signal = wavfile.read(os.path.join(folder, name))
            if rate != 16000:
                print(f"Skipping {name}: sample rate is {rate}, expected 16000")
                continue
            if signal.ndim > 1:          # stereo -> mono
                signal = signal[:, 0]
            X.append(extract_features(align_clip(signal)))
            y.append(label)
        print(f"{label:8s} {len(files)} files")
    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()
    if len(set(y)) < 2:
        raise SystemExit("Not enough data. Record samples first.")
    print(f"\nTotal: {len(y)} samples, {X.shape[1]} features each")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scaling matters for SVM: MFCC coefficients have very different ranges.
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, pred):.3f}\n")
    print(classification_report(y_test, pred))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=LABELS)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    # Final model: retrain on ALL data before saving.
    model.fit(X, y)
    joblib.dump({"model": model, "labels": LABELS}, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
