#!/usr/bin/env python3
"""Generator data sintetis landmark — *real-anchored*, bukan dari nol.

Metode (dipilih dari praktik komunitas, lihat docs/DATA.md §7b):

  mixup — interpolasi linear DI DALAM kelas antara dua sampel nyata yang
          berdekatan. Sederhana, aman, label tidak ambigu.
  pca   — model PCA per kelas + sampling Gaussian diagonal di ruang komponen
          utama, lalu inverse transform. Menghasilkan pose baru yang tetap
          berada di "kerucut variasi" kelas tersebut.
  both  — gabungan keduanya (default).

Prinsip yang ditegakkan skrip ini:

  1. REAL-ANCHORED — selalu dijalankan dari data nyata. Bukti komunitas:
     sintetis hanya membantu bila "anchored by real data" (arXiv 2605.09699).
  2. TIDAK UNTUK TEST — semua sampel ditandai signer="synthetic" sehingga
     train.py menempatkannya HANYA di train. Evaluasi wajib data nyata.
  3. BATAS DILUSI — maksimum `--max-fraction` × jumlah nyata per kelas
     (default 2×). Lebih dari itu, distribusi kelas didominasi halusinasi.
  4. SMOTE tidak dipakai — konsensus komunitas: tidak membantu classifier
     modern, merusak kalibrasi, dan lemah pada data kecil (lihat §7b).

Contoh:
    python scripts/generate_synthetic.py                      # both, 2× nyata
    python scripts/generate_synthetic.py --method mixup --per-class 150
    python scripts/generate_synthetic.py --dry-run            # cek statistik tanpa menulis
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import augment_features  # noqa: E402  (kebijakan augmentasi satu sumber)

SAMPLES_DIR = ROOT / "data" / "samples"
FEATURE_DIM = 63

METHODS = ("mixup", "pca", "both")


# --------------------------------------------------------------------------- #
# Pemuat data
# --------------------------------------------------------------------------- #


def load_real() -> list[dict]:
    """Baca semua CSV; lewati baris yang sudah sintetis (signer *synth*)."""
    samples: list[dict] = []
    for path in sorted(SAMPLES_DIR.glob("*.csv")):
        with path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header or header[0] != "class":
                continue
            signer_idx = header.index("signer") if "signer" in header else None

            for row in reader:
                if len(row) < FEATURE_DIM + 1:
                    continue
                signer = row[signer_idx] if signer_idx is not None and signer_idx < len(row) else "unknown"
                if signer.lower().startswith("synth"):
                    continue  # jangan meniru tiruan
                try:
                    feats = np.array([float(v) for v in row[1 : FEATURE_DIM + 1]])
                except ValueError:
                    continue
                samples.append({"label": row[0], "f": feats})

    return samples


# --------------------------------------------------------------------------- #
# Metode generasi
# --------------------------------------------------------------------------- #


def generate_mixup(M: np.ndarray, n: int, rng: np.random.Generator, k_neighbors: int = 20) -> np.ndarray:
    """Interpolasi dalam kelas antar tetangga terdekat (λ ~ U(0.2, 0.8))."""
    if len(M) < 2:
        return np.empty((0, M.shape[1]))

    # Jarak antar semua pasangan → pilih pasangan yang berdekatan saja,
    # supaya interpolasi tidak "menjembatani" dua mode gerakan yang jauh.
    sq = ((M[:, None, :] - M[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(sq, np.inf)
    k = min(k_neighbors, len(M) - 1)
    neighbors = np.argsort(sq, axis=1)[:, :k]

    out = np.empty((n, M.shape[1]))
    for i in range(n):
        a = rng.integers(len(M))
        b = neighbors[a][rng.integers(k)]
        lam = rng.uniform(0.2, 0.8)
        out[i] = lam * M[a] + (1 - lam) * M[b]
    return out


def generate_pca(M: np.ndarray, n: int, rng: np.random.Generator, max_components: int = 20) -> np.ndarray:
    """Sampling dari model PCA+Gaussian diagonal per kelas."""
    mu = M.mean(axis=0)
    centered = M - mu

    n_comp = int(min(max_components, len(M) - 1, M.shape[1]))
    if n_comp < 1:
        return np.empty((0, M.shape[1]))

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    comps = Vt[:n_comp]

    coeffs = centered @ comps.T                      # (n, n_comp)
    sigma = coeffs.std(axis=0, ddof=1) if len(M) > 1 else np.zeros(n_comp)
    sigma = np.maximum(sigma, 1e-6) * 0.9            # sedikit lebih rapat dari aslinya

    sampled = mu + (rng.normal(size=(n, n_comp)) * sigma) @ comps
    return sampled


# --------------------------------------------------------------------------- #
# Kontrol kualitas
# --------------------------------------------------------------------------- #


def fidelity_report(real: dict[str, np.ndarray], gen: dict[str, np.ndarray]) -> dict:
    """Ukur apakah sampel sintetis masih 'milik' kelasnya.

    - label match: centroid kelas sintetis terdekat harus kelas yang dimaksud
    - rasio jarak: median jarak-ke-centroid sintetis vs nyata (≈1 = wajar)
    """
    centroids = {c: M.mean(axis=0) for c, M in real.items()}
    labels = list(centroids)
    C = np.vstack([centroids[c] for c in labels])

    match, ratios = 0, []
    total = 0
    for c, G in gen.items():
        if len(G) == 0:
            continue
        # Label match
        d = np.sqrt(((G[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
        nearest = np.array(labels)[d.argmin(axis=1)]
        match += int((nearest == c).sum())
        total += len(G)

        # Rasio jarak
        rc = centroids[c]
        d_gen = np.sqrt(((G - rc) ** 2).sum(axis=1)).mean()
        d_real = np.sqrt(((real[c] - rc) ** 2).sum(axis=1)).mean()
        ratios.append(d_gen / d_real if d_real > 0 else float("nan"))

    return {
        "label_match_rate": round(match / total, 4) if total else 0.0,
        "distance_ratio_mean": round(float(np.nanmean(ratios)), 3) if ratios else None,
    }


# --------------------------------------------------------------------------- #
# Program utama
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="both")
    parser.add_argument("--per-class", type=int, default=0,
                        help="Jumlah sampel sintetis per kelas (0 = ikuti --max-fraction)")
    parser.add_argument("--max-fraction", type=float, default=2.0,
                        help="Batas maksimum rasio sintetis:nyata per kelas (default 2×)")
    parser.add_argument("--min-real", type=int, default=20,
                        help="Kelas dengan data nyata di bawah ini dilewati (default 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="generated_synthetic.csv")
    parser.add_argument("--dry-run", action="store_true", help="Laporkan saja, tidak menulis file")
    args = parser.parse_args()

    samples = load_real()
    if not samples:
        raise SystemExit("Tidak ada data nyata untuk dijadikan anchor.")

    per_class_real: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in samples:
        per_class_real[s["label"]].append(s["f"])
    real = {c: np.vstack(v) for c, v in per_class_real.items()}

    counts = Counter({c: len(M) for c, M in real.items()})
    print(f"Data nyata: {len(samples)} sampel, {len(real)} kelas — "
          f"{min(counts.values())}–{max(counts.values())}/kelas")

    rng = np.random.default_rng(args.seed)
    gen: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    for cls, M in sorted(real.items()):
        if len(M) < args.min_real:
            skipped.append(f"{cls}({len(M)})")
            continue

        target = args.per_class if args.per_class > 0 else int(len(M) * args.max_fraction)
        target = min(target, int(len(M) * args.max_fraction))

        parts = []
        if args.method in ("mixup", "both"):
            parts.append(generate_mixup(M, target // 2 if args.method == "both" else target, rng))
        if args.method in ("pca", "both"):
            parts.append(generate_pca(M, target - sum(len(p) for p in parts), rng))

        G = np.vstack([p for p in parts if len(p)])
        # Jitter prosedural ringan agar tidak berada persis di manifold linear.
        G = np.vstack([augment_features(g, rng) for g in G]) if len(G) else G
        gen[cls] = G

    total_gen = int(sum(len(G) for G in gen.values()))
    if not total_gen:
        raise SystemExit("Tidak ada sampel yang dihasilkan (semua kelas di bawah --min-real?).")

    report = fidelity_report(real, gen)
    print(f"\nDihasilkan {total_gen} sampel sintetis:")
    for cls in sorted(gen):
        print(f"  {cls}: nyata {len(real[cls])} + sintetis {len(gen[cls])}")
    if skipped:
        print(f"  dilewati (data nyata < {args.min_real}): {', '.join(skipped)}")

    print("\nKontrol kualitas:")
    print(f"  label sintetis cocok dengan kelasnya : {report['label_match_rate'] * 100:.1f}% (target ≥ 95%)")
    print(f"  rasio jarak sintetis/nyata           : {report['distance_ratio_mean']} (target ≈ 1.0)")
    if report["label_match_rate"] < 0.95 or (report["distance_ratio_mean"] or 0) > 1.5:
        print("  ⚠ Kualitas sintetis di bawah ambang — perlakukan sebagai eksperimen:")
        print("    jalankan scripts/benchmark.py dan hanya pakai bila terbukti menaikkan")
        print("    akurasi pada data nyata. Bila tidak, hapus file ini.")

    if args.dry_run:
        print("\n(dry-run: file tidak ditulis)")
        return 0

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLES_DIR / args.output
    session = f"synth-{args.method}-{datetime.now().strftime('%Y%m%d')}"
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class"] + [f"f{i}" for i in range(FEATURE_DIM)] + ["session", "signer"])
        for cls in sorted(gen):
            for feats in gen[cls]:
                writer.writerow([cls] + feats.tolist() + [session, "synthetic"])

    print(f"\nDitulis ke {out_path} (signer='synthetic', session='{session}').")
    print("Latih dengan: python scripts/train.py --include-synthetic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
