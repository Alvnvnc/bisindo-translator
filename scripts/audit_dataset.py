#!/usr/bin/env python3
"""Audit kualitas dataset landmark SEBELUM training.

Menjalankan checklist teori pembuatan data (lihat docs/DATA.md):

  1. Distribusi kelas     — jumlah sampel & ketidakseimbangan
  2. Cakupan metadata     — jumlah periset/sesi per kelas (untuk split SI)
  3. Kewarasan fitur      — NaN, rentang nilai, pergelangan di origin
  4. Duplikat             — sampel yang nyaris identik (inflasi akurasi palsu)
  5. Outlier per kelas    — sampel menyimpang (kemungkinan salah label)
  6. Risiko tertukar      — pasangan kelas yang centroid-nya berdekatan

Contoh:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --min-per-class 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
REPORTS_DIR = ROOT / "reports"
FEATURE_DIM = 63


def load_all() -> list[dict]:
    """Baca semua CSV → daftar sampel dengan metadata."""
    files = sorted(SAMPLES_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"Belum ada CSV di {SAMPLES_DIR}.")

    samples: list[dict] = []
    for path in files:
        with path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header or header[0] != "class":
                print(f"  ! {path.name} dilewati (header tidak dikenali)")
                continue

            session_idx = header.index("session") if "session" in header else None
            signer_idx = header.index("signer") if "signer" in header else None

            for line_no, row in enumerate(reader, start=2):
                if len(row) < FEATURE_DIM + 1:
                    continue
                try:
                    feats = np.array([float(v) for v in row[1 : FEATURE_DIM + 1]])
                except ValueError:
                    continue
                samples.append({
                    "label": row[0],
                    "f": feats,
                    "session": row[session_idx] if session_idx is not None and session_idx < len(row) else "unknown",
                    "signer": row[signer_idx] if signer_idx is not None and signer_idx < len(row) else "unknown",
                    "source": f"{path.name}:{line_no}",
                })

    if not samples:
        raise SystemExit("Tidak ada sampel valid.")
    return samples


def check_distribution(samples: list[dict], min_per_class: int) -> dict:
    counts = Counter(s["label"] for s in samples)
    values = sorted(counts.values())
    ratio = (max(values) / min(values)) if min(values) else float("inf")
    lowest = [f"{k}({v})" for k, v in sorted(counts.items(), key=lambda kv: kv[1])[:5]]

    return {
        "n_per_class": dict(sorted(counts.items())),
        "min": int(min(values)),
        "max": int(max(values)),
        "imbalance_ratio": round(float(ratio), 2),
        "below_min": [k for k, v in counts.items() if v < min_per_class],
        "lowest_classes": lowest,
    }


def check_metadata(samples: list[dict]) -> dict:
    signers = {s["signer"] for s in samples if s["signer"] not in ("", "unknown")}
    sessions = {s["session"] for s in samples if s["session"] not in ("", "unknown")}

    per_class_signers: dict[str, set] = defaultdict(set)
    per_class_sessions: dict[str, set] = defaultdict(set)
    for s in samples:
        if s["signer"] not in ("", "unknown"):
            per_class_signers[s["label"]].add(s["signer"])
        if s["session"] not in ("", "unknown"):
            per_class_sessions[s["label"]].add(s["session"])

    thin_classes = [k for k, v in per_class_signers.items() if len(v) < 2]
    no_meta = sum(1 for s in samples if s["signer"] in ("", "unknown") and s["session"] in ("", "unknown"))

    return {
        "n_signers": len(signers),
        "n_sessions": len(sessions),
        "signers": sorted(signers),
        "classes_with_lt2_signers": sorted(thin_classes),
        "n_samples_without_metadata": no_meta,
        "coverage": round(1 - no_meta / len(samples), 3),
    }


def check_sanity(samples: list[dict]) -> dict:
    X = np.vstack([s["f"] for s in samples])
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    finite = X[np.isfinite(X).all(axis=1)]

    # Setelah normalisasi, pergelangan (3 fitur pertama) harus di origin —
    # jadi wajar bila ketiganya konstan nol. Itu bukan masalah.
    wrist = np.abs(finite[:, :3]).sum(axis=1) if len(finite) else np.array([0.0])

    # Fitur selain pergelangan yang konstan di seluruh dataset tidak membawa informasi.
    std = finite.std(axis=0) if len(finite) else np.zeros(FEATURE_DIM)
    constant_body_features = int((std[3:] < 1e-9).sum())

    return {
        "n_nan": n_nan,
        "n_inf": n_inf,
        "max_abs_value": round(float(np.abs(finite).max()), 4) if len(finite) else 0.0,
        "wrist_origin_violations": int((wrist > 1e-6).sum()),
        "constant_body_features": constant_body_features,
    }


def check_duplicates(samples: list[dict], threshold: float) -> dict:
    """Hitung pasangan sampel nyaris identik DI DALAM kelas yang sama.

    Duplikat antar kelas justru indikasi salah label — itu dilaporkan terpisah.
    """
    per_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in samples:
        per_class[s["label"]].append(s["f"])

    within_pairs = 0
    worst_classes = []
    for label, feats in per_class.items():
        M = np.vstack(feats)
        if len(M) < 2:
            continue
        # Jarak kuadrat antar semua pasangan, kuadran atas saja.
        sq = ((M[:, None, :] - M[None, :, :]) ** 2).sum(axis=2)
        iu = np.triu_indices(len(M), k=1)
        d = np.sqrt(sq[iu])
        n_dup = int((d < threshold).sum())
        within_pairs += n_dup
        if n_dup:
            worst_classes.append((label, n_dup, float(d.min())))

    worst_classes.sort(key=lambda t: -t[1])
    return {
        "threshold": threshold,
        "duplicate_pairs_within_class": within_pairs,
        "top_classes": [
            {"class": c, "pairs": n, "min_distance": round(md, 6)} for c, n, md in worst_classes[:10]
        ],
    }


def check_outliers(samples: list[dict], z_limit: float) -> dict:
    """Sampel yang jaraknya ke centroid kelasnya ekstrem → kandidat salah label."""
    per_class: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        per_class[s["label"]].append(i)

    flagged: list[dict] = []
    for label, idxs in per_class.items():
        if len(idxs) < 10:
            continue
        M = np.vstack([samples[i]["f"] for i in idxs])
        centroid = M.mean(axis=0)
        d = np.sqrt(((M - centroid) ** 2).sum(axis=1))
        mu, sd = d.mean(), d.std()
        if sd < 1e-9:
            continue
        z = (d - mu) / sd
        for local_i in np.where(z > z_limit)[0]:
            flagged.append({
                "class": label,
                "source": samples[idxs[local_i]]["source"],
                "z": round(float(z[local_i]), 2),
            })

    flagged.sort(key=lambda r: -r["z"])
    return {"z_limit": z_limit, "n_flagged": len(flagged), "top": flagged[:15]}


def check_confusability(samples: list[dict]) -> dict:
    """Pasangan kelas dengan centroid terdekat — paling berisiko tertukar."""
    labels = sorted({s["label"] for s in samples})
    centroids = {}
    for label in labels:
        M = np.vstack([s["f"] for s in samples if s["label"] == label])
        centroids[label] = M.mean(axis=0)

    pairs = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            d = float(np.sqrt(((centroids[a] - centroids[b]) ** 2).sum()))
            pairs.append({"pair": f"{a} ↔ {b}", "distance": round(d, 4)})

    pairs.sort(key=lambda p: p["distance"])
    return {"closest_pairs": pairs[:10]}


def check_synthetic(samples: list[dict]) -> dict:
    """Ringkas porsi data sintetis — pengingat aturan: sintetis tidak untuk test."""
    per_class = defaultdict(lambda: {"real": 0, "synth": 0})
    for s in samples:
        key = "synth" if s["signer"].lower().startswith("synth") else "real"
        per_class[s["label"]][key] += 1

    n_synth = sum(v["synth"] for v in per_class.values())
    n_real = sum(v["real"] for v in per_class.values())
    over = [k for k, v in per_class.items() if v["real"] > 0 and v["synth"] > 2 * v["real"]]
    orphan = [k for k, v in per_class.items() if v["real"] == 0 and v["synth"] > 0]

    return {
        "n_real": n_real,
        "n_synthetic": n_synth,
        "share": round(n_synth / max(1, n_synth + n_real), 3),
        "over_diluted_classes": sorted(over),
        "classes_without_real_data": sorted(orphan),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-per-class", type=int, default=100,
                        help="Ambang minimal sampel per kelas (default 100)")
    parser.add_argument("--near-dup-threshold", type=float, default=1e-3,
                        help="Ambang jarak untuk dianggap duplikat (default 1e-3)")
    parser.add_argument("--outlier-z", type=float, default=3.5,
                        help="Ambang z-score outlier per kelas (default 3.5)")
    args = parser.parse_args()

    samples = load_all()
    synth = check_synthetic(samples)
    real_samples = [s for s in samples if not s["signer"].lower().startswith("synth")]

    summary = f"Memuat {len(samples)} sampel"
    if synth["n_synthetic"]:
        summary += f" ({synth['n_real']} nyata + {synth['n_synthetic']} sintetis)"
    print(summary + ".\n")

    if not real_samples:
        raise SystemExit("Hanya ada data sintetis — tidak ada yang bisa diaudit. Rekam data nyata dulu.")

    dist = check_distribution(real_samples, args.min_per_class)
    meta = check_metadata(real_samples)
    sanity = check_sanity(real_samples)
    dups = check_duplicates(real_samples, args.near_dup_threshold)
    outliers = check_outliers(real_samples, args.outlier_z)
    confus = check_confusability(real_samples)

    line = "─" * 62
    print(line)
    print("1. DISTRIBUSI KELAS")
    print(f"   {len(dist['n_per_class'])} kelas · {dist['min']}–{dist['max']} sampel/kelas "
          f"· rasio ketidakseimbangan {dist['imbalance_ratio']}×")
    if dist["below_min"]:
        print(f"   ⚠ < {args.min_per_class} sampel: {', '.join(dist['below_min'])}")
    if dist["imbalance_ratio"] > 3:
        print(f"   ⚠ Kelas timpang — tambah data untuk: {', '.join(dist['lowest_classes'])}")

    print(line)
    print("2. CAKUPAN METADATA (untuk split signer-independent)")
    print(f"   {meta['n_signers']} periset · {meta['n_sessions']} sesi · cakupan {meta['coverage'] * 100:.0f}%")
    if meta["n_signers"] < 3:
        print(f"   ⚠ Baru {meta['n_signers']} periset — target ≥ 3 agar split SI bermakna")
    if meta["classes_with_lt2_signers"]:
        print(f"   ⚠ Kelas dengan < 2 periset: {', '.join(meta['classes_with_lt2_signers'])}")

    print(line)
    print("3. KEWARASAN FITUR")
    problems = []
    if sanity["n_nan"] or sanity["n_inf"]:
        problems.append(f"{sanity['n_nan']} NaN, {sanity['n_inf']} Inf")
    if sanity["wrist_origin_violations"]:
        problems.append(f"{sanity['wrist_origin_violations']} sampel bukan dari normalisasi yang benar")
    if sanity["constant_body_features"]:
        problems.append(f"{sanity['constant_body_features']} fitur tubuh konstan")
    print(f"   rentang nilai maks |v| = {sanity['max_abs_value']}")
    print("   ✓ tidak ada masalah" if not problems else f"   ⚠ {', '.join(problems)}")

    print(line)
    print("4. DUPLIKAT (sampel nyaris identik dalam kelas yang sama)")
    print(f"   {dups['duplicate_pairs_within_class']} pasangan di bawah ambang {dups['threshold']}")
    if dups["top_classes"]:
        top = ", ".join(f"{c['class']}({c['pairs']})" for c in dups["top_classes"][:5])
        print(f"   ⚠ Kelas terbanyak: {top}")
        print("   → Hapus duplikat: data kembar membuat akurasi test terlihat bagus palsu.")

    print(line)
    print("5. OUTLIER (kandidat salah label / posisi aneh)")
    print(f"   {outliers['n_flagged']} sampel dengan z > {outliers['z_limit']}")
    for o in outliers["top"][:5]:
        print(f"   - {o['class']} z={o['z']} ({o['source']})")

    print(line)
    print("6. RISIKO TERTUKAR (jarak centroid antar kelas)")
    for p in confus["closest_pairs"][:5]:
        print(f"   {p['pair']}: {p['distance']}")
    print("   → Pasangan terdekat adalah prioritas perbaikan: tambah variasi data")
    print("     atau perjelas definisi isyarat di docs/DATA.md.")

    if synth["n_synthetic"]:
        print(line)
        print("7. DATA SINTETIS (aturan: hanya untuk train, test wajib nyata)")
        print(f"   {synth['n_synthetic']} sintetis vs {synth['n_real']} nyata "
              f"({synth['share'] * 100:.0f}% sintetis)")
        if synth["over_diluted_classes"]:
            print(f"   ⚠ Sintetis > 2× nyata di: {', '.join(synth['over_diluted_classes'])}")
        if synth["classes_without_real_data"]:
            print(f"   ⚠ Kelas tanpa data nyata: {', '.join(synth['classes_without_real_data'])} "
                  "— tidak akan pernah teruji")
    print(line)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"data_audit_{stamp}.json"
    out.write_text(json.dumps({
        "timestamp": stamp,
        "n_samples": len(samples),
        "distribution": dist,
        "metadata": meta,
        "sanity": sanity,
        "duplicates": dups,
        "outliers": outliers,
        "confusability": confus,
        "synthetic": synth,
    }, indent=2))
    print(f"Laporan lengkap: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
