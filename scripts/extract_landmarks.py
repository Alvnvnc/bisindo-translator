#!/usr/bin/env python3
"""Ekstrak landmark tangan dari gambar/video mentah menjadi CSV sampel.

Input : data/raw/<KELAS>/*.{jpg,jpeg,png,webp,mp4,mov}
Output: data/samples/raw_<sumber>.csv  dengan kolom  class,f0..f62

Format CSV ini identik dengan yang dihasilkan web/capture.html, sehingga kedua
sumber data bisa digabung langsung untuk training.

Contoh:
    python scripts/extract_landmarks.py                  # proses semua kelas
    python scripts/extract_landmarks.py --classes A B C  # hanya kelas tertentu
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SAMPLES_DIR = ROOT / "data" / "samples"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm"}

# 21 landmark MediaPipe Hand Landmarker.
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # x, y, z


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalisasi landmark agar invarian terhadap posisi & ukuran tangan.

    PENTING: implementasi ini harus identik dengan normalizeLandmarks() di
    web/landmarks.js, kalau tidak model hasil training tidak akan bekerja di
    browser.

    Args:
        landmarks: array (21, 3) koordinat ternormalisasi [0,1] dari MediaPipe.

    Returns:
        array (21, 3) yang sudah digeser ke pergelangan tangan dan diskalakan.
    """
    wrist = landmarks[0]
    translated = landmarks - wrist

    # Skala = jarak terjauh pergelangan → landmark mana pun.
    distances = np.linalg.norm(translated, axis=1)
    scale = float(distances.max())
    if scale < 1e-6:
        scale = 1.0

    return translated / scale


def ensure_model() -> Path:
    """Unduh model hand_landmarker.task bila belum ada."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh model MediaPipe ke {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Selesai.")
    return MODEL_PATH


def build_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    base_options = mp_tasks.BaseOptions(model_asset_path=str(ensure_model()))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    return mp, vision.HandLandmarker.create_from_options(options)


def extract_from_image(path: Path, landmarker, mp) -> np.ndarray | None:
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"  ! Gagal membaca gambar: {path.name}")
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(image)

    if not result.hand_landmarks:
        return None

    lms = result.hand_landmarks[0]
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float64)
    if arr.shape != (NUM_LANDMARKS, 3):
        return None
    return normalize_landmarks(arr)


def extract_from_video(path: Path, landmarker_factory, mp, stride: int = 5):
    """Ambil sampel landmark dari video setiap `stride` frame."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ! Gagal membuka video: {path.name}")
        return

    landmarker = landmarker_factory()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(image)
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float64)
                if arr.shape == (NUM_LANDMARKS, 3):
                    yield normalize_landmarks(arr)
        frame_idx += 1

    cap.release()
    landmarker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="*", help="Batasi kelas tertentu")
    parser.add_argument("--stride", type=int, default=5, help="Ambil 1 dari N frame video")
    parser.add_argument("--signer", default="unknown",
                        help="ID periset untuk seluruh file (dipakai bila folder tidak berisi subfolder periset)")
    parser.add_argument("--output", default="raw_dataset.csv", help="Nama file CSV output")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"Folder {RAW_DIR} belum ada. Buat struktur data/raw/<KELAS>/ dulu.")
        return 1

    class_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    if args.classes:
        wanted = {c.upper() for c in args.classes}
        class_dirs = [d for d in class_dirs if d.name.upper() in wanted]

    if not class_dirs:
        print("Tidak ada folder kelas yang cocok di data/raw/")
        return 1

    mp, landmarker = build_landmarker()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLES_DIR / args.output
    # Kolom tambahan session & signer wajib ada agar split signer-independent
    # bisa dilakukan (lihat docs/DATA.md).
    header = ["class"] + [f"f{i}" for i in range(FEATURE_DIM)] + ["session", "signer"]

    total = 0
    skipped = 0
    counts: dict[str, int] = {}

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for class_dir in class_dirs:
            label = class_dir.name
            files = sorted(p for p in class_dir.rglob("*") if p.is_file())
            for path in files:
                suffix = path.suffix.lower()

                # Konvensi opsional: data/raw/<KELAS>/<SIGNER>/file → signer dari subfolder.
                rel = path.relative_to(class_dir)
                signer = rel.parts[0] if len(rel.parts) > 1 else args.signer
                session = f"{signer}:{path.stem}"

                if suffix in IMAGE_EXTS:
                    feats = extract_from_image(path, landmarker, mp)
                    if feats is None:
                        skipped += 1
                        continue
                    writer.writerow([label] + feats.flatten().tolist() + [session, signer])
                    counts[label] = counts.get(label, 0) + 1
                    total += 1

                elif suffix in VIDEO_EXTS:
                    for feats in extract_from_video(path, lambda: build_landmarker()[1], mp, args.stride):
                        writer.writerow([label] + feats.flatten().tolist() + [session, signer])
                        counts[label] = counts.get(label, 0) + 1
                        total += 1

                else:
                    skipped += 1

            print(f"  {label}: {counts.get(label, 0)} sampel")

    landmarker.close()

    print(f"\nTotal {total} sampel ditulis ke {out_path}")
    if skipped:
        print(f"({skipped} file dilewati: tidak terdeteksi tangan atau format tidak didukung)")
    if not total:
        print("Belum ada sampel. Pastikan ada gambar/video di data/raw/<KELAS>/.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
