#!/usr/bin/env python3
"""Uji paritas Python ↔ JavaScript.

Menjaga invariant terpenting proyek ini: normalisasi landmark dan inference
classifier di browser HARUS menghasilkan angka yang sama dengan Python.
Kalau tidak, model yang dilatih di Python tidak akan berfungsi di web demo.

    python scripts/check_parity.py

Butuh Node.js di PATH dan web/model.json hasil training.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_landmarks import normalize_landmarks  # noqa: E402

MODEL_PATH = ROOT / "web" / "model.json"
SAMPLES_GLOB = ROOT / "data" / "samples" / "*.csv"
JS_RUNNER = Path(__file__).resolve().parent / "check_parity.mjs"


def build_norm_cases(rng: np.random.Generator, n: int = 25) -> list[dict]:
    cases = []
    for _ in range(n):
        lms = rng.normal(0.5, 0.2, size=(21, 3))
        lms[0] = rng.normal(0.5, 0.2, size=3)
        norm = normalize_landmarks(lms)
        cases.append({
            "input": [{"x": float(l[0]), "y": float(l[1]), "z": float(l[2])} for l in lms],
            "expected": norm.flatten().tolist(),
        })
    return cases


def load_feature_rows(limit: int = 100) -> np.ndarray:
    rows: list[list[str]] = []
    for path in sorted(SAMPLES_GLOB.parent.glob("*.csv")):
        with path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) == 64:
                    rows.append(row[1:])
    if not rows:
        raise SystemExit("Belum ada CSV di data/samples/ — jalankan make_sample_dataset.py dulu.")
    rng = np.random.default_rng(1)
    idx = rng.choice(len(rows), size=min(limit, len(rows)), replace=False)
    return np.array([[float(v) for v in rows[i]] for i in idx])


def expected_predictions(model: dict, X: np.ndarray) -> list[dict]:
    mean = np.array(model["scaler"]["mean"])
    scale = np.array(model["scaler"]["scale"])
    Xs = (X - mean) / scale
    out = []

    if model["type"] == "knn":
        S = np.array(model["samples"])
        L = np.array(model["labels"])
        k = int(model["k"])
        for i, x in enumerate(Xs):
            d = ((S - x) ** 2).sum(axis=1)
            idx = np.argsort(d, kind="stable")[:k]
            votes = np.bincount(L[idx], minlength=len(model["classes"]))
            pred = int(np.argmax(votes))
            out.append({
                "features": X[i].tolist(),
                "index": pred,
                "label": model["classes"][pred],
                "confidence": float(votes[pred] / k),
            })
    elif model["type"] == "mlp":
        vec = Xs
        last = len(model["layers"]) - 1
        for li, layer in enumerate(model["layers"]):
            z = vec @ np.array(layer["weights"]) + np.array(layer["bias"])
            vec = z if li == last else np.maximum(z, 0)
        if model["out_activation"] == "softmax":
            e = np.exp(vec - vec.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
        else:
            probs = vec
        preds = probs.argmax(axis=1)
        for i in range(len(X)):
            p = int(preds[i])
            out.append({
                "features": X[i].tolist(),
                "index": p,
                "label": model["classes"][p],
                "confidence": float(probs[i][p]),
            })
    else:
        raise SystemExit(f"Tipe model tidak dikenali: {model['type']}")

    return out


def main() -> int:
    if not MODEL_PATH.exists():
        raise SystemExit("web/model.json belum ada — jalankan scripts/train.py dulu.")

    model = json.loads(MODEL_PATH.read_text())
    rng = np.random.default_rng(0)
    cases = {
        "model_type": model["type"],
        "norm": build_norm_cases(rng),
        "predict": expected_predictions(model, load_feature_rows()),
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cases, fh)
        cases_path = Path(fh.name)

    print(f"Menguji paritas untuk model '{model['type']}' "
          f"({len(cases['norm'])} kasus normalisasi, {len(cases['predict'])} kasus prediksi)…\n")

    result = subprocess.run(
        ["node", str(JS_RUNNER), str(cases_path), str(ROOT / "web" / "landmarks.js")],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.stderr.strip():
        print(result.stderr, end="")

    cases_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print("\n✗ Paritas GAGAL — jangan lanjut mengembangkan fitur sebelum ini beres.")
        return 1

    print("\n✓ Paritas OK — Python dan browser menghitung hal yang sama.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
