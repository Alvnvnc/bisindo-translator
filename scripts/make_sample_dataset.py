#!/usr/bin/env python3
"""Buat dataset sintetis untuk smoke-test pipeline training.

Ini BUKAN data isyarat asli — hanya untuk memastikan alur
`data/samples/*.csv → scripts/train.py → web/model.json` berjalan sebelum
dataset sungguhan tersedia.

    python scripts/make_sample_dataset.py
    python scripts/train.py --model knn
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_landmarks import FEATURE_DIM, NUM_LANDMARKS, normalize_landmarks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples" / "synthetic.csv"

CLASSES = ["A", "B", "C", "D", "E"]
SAMPLES_PER_CLASS = 250
SEED = 7


def rotation_matrix(rng: np.random.Generator, max_deg: float = 12.0) -> np.ndarray:
    """Rotasi acak kecil di sekitar tiga sumbu (variasi sudut tangan)."""
    angles = rng.uniform(-max_deg, max_deg, 3) * np.pi / 180
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def prototype(rng: np.random.Generator) -> np.ndarray:
    """Bentuk tangan acak yang konsisten untuk satu kelas."""
    base = rng.normal(0.0, 0.35, size=(NUM_LANDMARKS, 3))
    base[0] = 0.0  # pergelangan di origin
    return base


def main() -> int:
    rng = np.random.default_rng(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    prototypes = {c: prototype(rng) for c in CLASSES}

    rows: list[list] = []
    for cls in CLASSES:
        proto = prototypes[cls]
        for _ in range(SAMPLES_PER_CLASS):
            rotated = proto @ rotation_matrix(rng).T
            noisy = rotated + rng.normal(0.0, 0.02, size=rotated.shape)
            rows.append([cls] + normalize_landmarks(noisy).flatten().tolist())

    rng.shuffle(rows)

    with OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class"] + [f"f{i}" for i in range(FEATURE_DIM)])
        writer.writerows(rows)

    print(f"{len(rows)} sampel sintetis ({len(CLASSES)} kelas) → {OUT}")
    print("Jalankan: python scripts/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
