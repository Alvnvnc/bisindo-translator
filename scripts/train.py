#!/usr/bin/env python3
"""Latih classifier isyarat tangan dan ekspor ke web/model.json.

Membaca semua CSV di data/samples/, melatih beberapa kandidat model, memilih
yang terbaik berdasarkan cross-validation, lalu:
  1. mengevaluasi pada hold-out set (accuracy + laporan per kelas),
  2. mengekspor model terbaik ke web/model.json (dipakai langsung oleh browser),
  3. menyimpan metrik ke reports/.

Contoh:
    python scripts/train.py                  # otomatis pilih model terbaik
    python scripts/train.py --model knn      # paksa KNN
    python scripts/train.py --model mlp      # paksa MLP
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
REPORTS_DIR = ROOT / "reports"
MODEL_OUT = ROOT / "web" / "model.json"

FEATURE_DIM = 63


def load_samples() -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[str]]]:
    """Gabungkan semua CSV di data/samples/ menjadi (X, y, classes, meta).

    Kolom wajib: class,f0..f62. Kolom opsional (dibaca berdasarkan nama header):
    session dan signer — dipakai untuk split signer-independent.
    """
    files = sorted(SAMPLES_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"Belum ada data di {SAMPLES_DIR}.\n"
            "Rekam dulu lewat web/capture.html, atau jalankan scripts/extract_landmarks.py."
        )

    labels: list[str] = []
    features: list[list[float]] = []
    meta: dict[str, list[str]] = {"session": [], "signer": []}

    for path in files:
        with path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header or header[0] != "class":
                print(f"  ! {path.name} dilewati (format header tidak dikenali)")
                continue

            session_idx = header.index("session") if "session" in header else None
            signer_idx = header.index("signer") if "signer" in header else None

            for row in reader:
                if len(row) < FEATURE_DIM + 1:
                    print(f"  ! baris rusak di {path.name} dilewati ({len(row)} kolom)")
                    continue
                try:
                    feats = [float(v) for v in row[1 : FEATURE_DIM + 1]]
                except ValueError:
                    print(f"  ! baris non-numerik di {path.name} dilewati")
                    continue

                labels.append(row[0])
                features.append(feats)
                meta["session"].append(
                    row[session_idx] if session_idx is not None and session_idx < len(row) else "unknown"
                )
                meta["signer"].append(
                    row[signer_idx] if signer_idx is not None and signer_idx < len(row) else "unknown"
                )

    if not features:
        raise SystemExit("Tidak ada sampel valid yang terbaca.")

    X = np.array(features, dtype=np.float64)
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])

    print(f"Memuat {len(features)} sampel dari {len(files)} file, {len(classes)} kelas.")
    return X, y, classes, meta


def build_candidates() -> dict[str, object]:
    """Kandidat model. Semua dibungkus StandardScaler agar konsisten."""
    return {
        "knn": make_pipeline(
            StandardScaler(),
            GridSearchCV(
                KNeighborsClassifier(),
                {"n_neighbors": [1, 3, 5, 7, 9]},
                cv=3,
                n_jobs=-1,
            ),
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                max_iter=600,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
            ),
        ),
        "rf": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    }


def cv_score(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> float:
    """Cross-validation; menghormati batas grup (periset/sesi) bila diberikan.

    Tanpa `groups`, sampel dari orang yang sama bisa masuk train dan test
    sekaligus — akurasi jadi terlalu optimistis. Lihat docs/DATA.md.
    """
    n_splits = min(5, min(Counter(y).values()))
    if groups is not None:
        n_splits = min(n_splits, len(set(groups.tolist())))
    if n_splits < 2:
        return 0.0

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splitter = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splitter = cv.split(X, y)

    scores = []
    for train_idx, test_idx in splitter:
        model.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    return float(np.mean(scores))


def make_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    test_size: float,
    seed: int = 42,
) -> tuple:
    """Bagi data menjadi train/test.

    Bila `groups` diberikan (mis. ID periset), seluruh grup hanya muncul di salah
    satu sisi — inilah protokol *signer-independent* yang mengukur generalisasi
    ke orang yang belum pernah dilihat model.
    """
    if groups is not None and len(set(groups.tolist())) >= 2:
        n_splits = min(max(2, round(1 / test_size)), len(set(groups.tolist())))
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            train_idx, test_idx = next(sgkf.split(X, y, groups))
            held = sorted(set(groups[test_idx].tolist()))
            info = {
                "protocol": "group-independent (StratifiedGroupKFold)",
                "test_groups": held,
                "n_groups_train": len(set(groups[train_idx].tolist())),
            }
        except ValueError as exc:
            print(f"  ! StratifiedGroupKFold gagal ({exc}); fallback GroupShuffleSplit")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(gss.split(X, y, groups))
            held = sorted(set(groups[test_idx].tolist()))
            info = {
                "protocol": "group-independent (GroupShuffleSplit)",
                "test_groups": held,
                "n_groups_train": len(set(groups[train_idx].tolist())),
            }
        return (
            X[train_idx], X[test_idx], y[train_idx], y[test_idx],
            groups[train_idx], groups[test_idx], info,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    info = {
        "protocol": "stratified acak — PERINGATAN: tidak mengukur generalisasi antar-individu",
        "test_groups": [],
        "n_groups_train": 0,
    }
    return X_train, X_test, y_train, y_test, None, None, info


def augment_features(
    feat: np.ndarray,
    rng: np.random.Generator,
    max_deg: float = 12.0,
    noise_sigma: float = 0.01,
) -> np.ndarray:
    """Augmentasi landmark sesuai teori (docs/DATA.md §7).

    Boleh: rotasi kecil (± 12°) + noise Gaussian pada landmark tubuh.
    Tidak boleh: translasi/skala (sudah invarian), flip horizontal (mengubah
    makna huruf), warp besar. Pergelangan dibiarkan tetap di origin.
    """
    pts = feat.reshape(21, 3).copy()

    angles = rng.uniform(-max_deg, max_deg, 3) * np.pi / 180.0
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pts = pts @ (rz @ ry @ rx).T

    pts[1:] += rng.normal(0.0, noise_sigma, size=(20, 3))
    return pts.flatten()


def augment_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray | None,
    copies: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Tambah `copies` salinan ter-augmentasi untuk tiap sampel training.

    Hanya dipanggil untuk data train — augmentasi pada data test adalah
    kebocoran data dan dilarang.
    """
    rng = np.random.default_rng(seed)
    X_parts = [X_train]
    y_parts = [y_train]
    g_parts = [g_train] if g_train is not None else None

    for _ in range(copies):
        X_parts.append(np.vstack([augment_features(f, rng) for f in X_train]))
        y_parts.append(y_train)
        if g_parts is not None:
            g_parts.append(g_train)

    groups = np.concatenate(g_parts) if g_parts is not None else None
    return np.vstack(X_parts), np.concatenate(y_parts), groups


# --------------------------------------------------------------------------- #
# Ekspor model ke JSON agar bisa dipakai langsung di browser (tanpa server ML)
# --------------------------------------------------------------------------- #


def export_knn(model, X_train: np.ndarray, y_train: np.ndarray, classes: list[str]) -> dict:
    """Serialisasi KNN: simpan seluruh vektor training + label indeks kelas."""
    scaler = model.steps[0][1]
    knn = model.steps[1][1]
    if hasattr(knn, "best_estimator_"):  # hasil GridSearchCV
        knn = knn.best_estimator_
    Xs = scaler.transform(X_train)
    return {
        "type": "knn",
        "classes": classes,
        "k": int(knn.n_neighbors),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "samples": Xs.round(6).tolist(),
        "labels": [int(l) for l in y_train],
    }


def export_mlp(model, classes: list[str]) -> dict:
    """Serialisasi MLP.

    sklearn menyimpan coefs_[i] dengan bentuk (n_in, n_out) — bentuk ini dipakai
    apa adanya agar cocok dengan mlpPredict() di web/landmarks.js.
    """
    scaler = model.steps[0][1]
    mlp = model.steps[1][1]
    layers = [
        {"weights": coef.tolist(), "bias": inter.tolist()}
        for coef, inter in zip(mlp.coefs_, mlp.intercepts_)
    ]
    return {
        "type": "mlp",
        "classes": classes,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "layers": layers,
        "out_activation": str(mlp.out_activation_),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["knn", "mlp", "rf", "auto"], default="auto")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--group-column", choices=["signer", "session", "none"], default="signer",
                        help="Kolom metadata untuk split group-independent (default: signer)")
    parser.add_argument("--export-train-set", action="store_true",
                        help="Ekspor model yang dilatih ulang pada seluruh data")
    parser.add_argument("--augment", type=int, default=0,
                        help="Jumlah salinan augmentasi (rotasi kecil + noise) per sampel training; 0 = mati")
    args = parser.parse_args()

    X, y, classes, meta = load_samples()

    counts = Counter(y.tolist())
    print("Distribusi kelas:", {classes[k]: v for k, v in sorted(counts.items())})
    if min(counts.values()) < 5:
        print("\n⚠ Ada kelas dengan < 5 sampel — cross-validation tidak bisa dipercaya.")
        print("  Tambah data dulu lewat web/capture.html.\n")

    # --- Tentukan grup untuk split (teori: uji pada individu yang belum dilihat) ---
    groups = None
    if args.group_column != "none":
        values = meta.get(args.group_column, [])
        valid = {v for v in values if v and v != "unknown"}
        if len(valid) >= 2:
            groups = np.array(values)
            shown = ", ".join(sorted(valid)[:8]) + ("…" if len(valid) > 8 else "")
            print(f"Split group-independent pada kolom '{args.group_column}': {len(valid)} grup ({shown})")
        else:
            print(f"⚠ Kolom '{args.group_column}' tidak punya ≥2 nilai valid — split acak dipakai; "
                  "akurasi akan terlalu optimistis. Rekam data dengan ID periset berbeda.")

    X_train, X_test, y_train, y_test, g_train, g_test, split_info = make_split(
        X, y, groups, args.test_size
    )
    print(f"Protokol split : {split_info['protocol']}")
    if split_info["test_groups"]:
        print(f"Grup uji       : {', '.join(split_info['test_groups'])} (tidak dilihat saat training)")

    # Augmentasi HANYA untuk train — test set tidak boleh disentuh (docs/DATA.md §7).
    if args.augment > 0:
        n_before = len(X_train)
        X_train, y_train, g_train = augment_training_set(X_train, y_train, g_train, args.augment)
        print(f"Augmentasi     : {args.augment}× (rotasi ±12° + noise) → "
              f"{n_before} menjadi {len(X_train)} sampel train; test tidak diaugmentasi")

    candidates = build_candidates()
    if args.model != "auto":
        candidates = {args.model: candidates[args.model]}

    results: dict[str, float] = {}
    fitted: dict[str, object] = {}
    for name, model in candidates.items():
        print(f"\nMelatih {name} ...")
        score = cv_score(model, X_train, y_train, g_train)
        model.fit(X_train, y_train)
        results[name] = score
        fitted[name] = model
        print(f"  CV accuracy: {score:.4f}")

    best_name = max(results, key=lambda k: results[k])
    best = fitted[best_name]
    print(f"\n=== Model terbaik: {best_name} (CV {results[best_name]:.4f}) ===")

    y_pred = best.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, labels=range(len(classes)), target_names=classes, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(report)

    worst = []
    for i, row in enumerate(cm):
        errors = row.copy()
        errors[i] = 0
        if errors.sum():
            j = int(np.argmax(errors))
            worst.append(f"  {classes[i]} sering tertukar dengan {classes[j]} ({errors[j]}×)")
    if worst:
        print("Kebingungan tersering:")
        print("\n".join(worst))

    # --- Ekspor untuk browser -------------------------------------------------
    if args.export_train_set:
        final = build_candidates()[best_name]
        final.fit(X, y)
        export_model, export_X, export_y = final, X, y
    else:
        export_model, export_X, export_y = best, X_train, y_train

    if best_name == "knn":
        payload = export_knn(export_model, export_X, export_y, classes)
    elif best_name == "mlp":
        payload = export_mlp(export_model, classes)
    else:
        payload = None  # RandomForest tidak bisa dijalankan langsung di browser

    if payload is None:
        print(f"\n⚠ Model {best_name} tidak bisa dijalankan di browser. "
              "Pakai --model knn atau --model mlp untuk ekspor.")
    else:
        payload.update({
            "meta": {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": best_name,
                "cv_accuracy": round(results[best_name], 4),
                "test_accuracy": round(float(test_acc), 4),
                "n_train": int(len(export_X)),
                "n_classes": len(classes),
                "split_protocol": split_info["protocol"],
                "test_groups": split_info["test_groups"],
            },
        })
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        MODEL_OUT.write_text(json.dumps(payload, separators=(",", ":")))
        size_kb = MODEL_OUT.stat().st_size / 1024
        print(f"\nModel diekspor ke {MODEL_OUT} ({size_kb:.0f} KB)")

    # --- Simpan laporan -------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        "timestamp": stamp,
        "cv_scores": {k: round(v, 4) for k, v in results.items()},
        "best_model": best_name,
        "test_accuracy": round(float(test_acc), 4),
        "classes": classes,
        "class_counts": {classes[k]: v for k, v in counts.items()},
        "split": split_info,
        "augment_copies": args.augment,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    report_path = REPORTS_DIR / f"train_{stamp}.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"Laporan disimpan ke {report_path}")

    if test_acc < 0.85:
        print("\n⚠ Akurasi di bawah target 85%. Sebelum lanjut ke UI:")
        print("   - tambah sampel per kelas (target ≥ 300, dari ≥ 3 periset)")
        print("   - rekam dengan variasi jarak, sudut, pencahayaan, latar")
        print("   - cek kelas yang paling sering tertukar di atas")
        print("   - jalankan scripts/audit_dataset.py untuk melihat masalah data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
