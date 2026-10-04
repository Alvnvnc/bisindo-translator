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
SIGNERS = ["S01", "S02", "S03", "S04", "S05"]
SAMPLES_PER_CLASS_PER_SIGNER = 50
SESSIONS_PER_PAIR = 2
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

    # Setiap periset punya "gaya" tersendiri + prototipe kelas, sehingga split
    # signer-independent benar-benar menguji generalisasi antar-individu.
    signer_bias = {s: rng.normal(0.0, 0.15, size=(NUM_LANDMARKS, 3)) for s in SIGNERS}
    prototypes = {c: prototype(rng) for c in CLASSES}

    rows: list[list] = []
    for signer in SIGNERS:
        for cls in CLASSES:
            proto = prototypes[cls] + signer_bias[signer]
            per_session = SAMPLES_PER_CLASS_PER_SIGNER // SESSIONS_PER_PAIR

            for session_idx in range(SESSIONS_PER_PAIR):
                session = f"{signer}-ses{session_idx + 1}"
                for _ in range(per_session):
                    rotated = proto @ rotation_matrix(rng).T
                    noisy = rotated + rng.normal(0.0, 0.02, size=rotated.shape)
                    feats = normalize_landmarks(noisy).flatten().tolist()
                    rows.append([cls] + feats + [session, signer])

    rng.shuffle(rows)

    with OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class"] + [f"f{i}" for i in range(FEATURE_DIM)] + ["session", "signer"])
        writer.writerows(rows)

    print(
        f"{len(rows)} sampel sintetis ({len(CLASSES)} kelas × {len(SIGNERS)} periset) → {OUT}"
    )
    print("Jalankan: python scripts/train.py  (atau --group-column signer untuk split SI)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
