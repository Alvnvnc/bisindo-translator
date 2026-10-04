#!/usr/bin/env python3
"""Benchmark terkontrol: baseline vs augmentasi vs data sintetis.

Semua konfigurasi dievaluasi pada SPLIT YANG SAMA (seed identik, test
signer-independent yang sama, test 100% data nyata) sehingga perbandingannya
sah secara metodologis — bukan empat eksperimen terpisah yang kebetulan beda
pembagian data.

Tidak menyentuh web/model.json (demo web tidak terganggu). Hasil:

  - tabel di terminal + rekomendasi
  - reports/benchmark_<stamp>.json  → data mentah lengkap
  - reports/benchmark_<stamp>.md    → tabel siap tempel ke dokumentasi Devpost

Contoh:
    python scripts/benchmark.py
    python scripts/benchmark.py --models knn --augment 3
    python scripts/benchmark.py --models knn,mlp,rf --cv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (  # noqa: E402
    augment_training_set,
    build_candidates,
    export_knn,
    export_mlp,
    load_samples,
    make_split,
    cv_score,
)

REPORTS_DIR = ROOT / "reports"
MARGINAL_GAIN = 0.005  # peningkatan < 0.5 poin dianggap marginal


def serialize_size_kb(model_name: str, fitted, X_fit: np.ndarray, y_fit: np.ndarray, classes: list[str]) -> float | None:
    """Ukuran model yang akan dikirim ke browser (tanpa menulis file)."""
    if model_name == "knn":
        payload = export_knn(fitted, X_fit, y_fit, classes)
    elif model_name == "mlp":
        payload = export_mlp(fitted, classes)
    else:
        return None  # rf tidak diekspor ke browser
    return round(len(json.dumps(payload, separators=(",", ":"))) / 1024, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="knn,mlp",
                        help="Model yang dibandingkan, pisahkan dengan koma (default knn,mlp)")
    parser.add_argument("--augment", type=int, default=2,
                        help="Jumlah salinan augmentasi untuk konfigurasi augmentasi (default 2)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", action="store_true",
                        help="Hitung juga CV pada data nyata (lebih lambat)")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_names if m not in ("knn", "mlp", "rf")]
    if unknown:
        raise SystemExit(f"Model tidak dikenali: {', '.join(unknown)}")

    X, y, classes, meta = load_samples()

    synth_mask = np.array([s.lower().startswith("synth") for s in meta["signer"]])
    n_synth = int(synth_mask.sum())
    has_synth = n_synth > 0

    keep_real = ~synth_mask
    X_real, y_real = X[keep_real], y[keep_real]
    meta_real = {k: [v for v, keep in zip(meta[k], keep_real.tolist()) if keep] for k in meta}

    if not has_synth:
        print("ℹ Tidak ada data sintetis — konfigurasi sintetis dilewati. "
              "Jalankan scripts/generate_synthetic.py bila ingin mengujinya.\n")

    groups = None
    values = meta_real.get("signer", [])
    valid = {v for v in values if v and v != "unknown"}
    if len(valid) >= 2:
        groups = np.array(values)

    # Satu split, dipakai SEMUA konfigurasi — kunci keadilan perbandingan.
    X_train, X_test, y_train, y_test, g_train, g_test, split_info = make_split(
        X_real, y_real, groups, args.test_size
    )

    print(f"Data nyata     : {len(X_real)} sampel, {len(classes)} kelas"
          + (f" (+{n_synth} sintetis tersedia)" if has_synth else ""))
    print(f"Protokol split : {split_info['protocol']}")
    if split_info["test_groups"]:
        print(f"Grup uji       : {', '.join(split_info['test_groups'])} — {len(X_test)} sampel nyata")
    print(f"Test set sama untuk semua konfigurasi. Linimasa: {datetime.now().strftime('%d %b %Y %H:%M')}\n")

    # --- Susun konfigurasi ----------------------------------------------------
    configs = [("baseline", False, 0)]
    if args.augment > 0:
        configs.append((f"augmentasi {args.augment}×", False, args.augment))
    if has_synth:
        configs.append(("sintetis", True, 0))
        if args.augment > 0:
            configs.append((f"sintetis + augmentasi {args.augment}×", True, args.augment))

    rows: list[dict] = []

    for cfg_name, use_synth, aug_copies in configs:
        # Bangun himpunan fitting untuk konfigurasi ini
        X_fit, y_fit, g_fit = X_train, y_train, g_train
        n_synth_used = 0
        if use_synth:
            X_syn, y_syn = X[synth_mask], y[synth_mask]
            X_fit = np.vstack([X_fit, X_syn])
            y_fit = np.concatenate([y_fit, y_syn])
            if g_fit is not None:
                g_fit = np.concatenate([g_fit, np.full(len(X_syn), "synthetic", dtype=object)])
            n_synth_used = len(X_syn)
        if aug_copies > 0:
            X_fit, y_fit, g_fit = augment_training_set(X_fit, y_fit, g_fit, aug_copies)

        print(f"▸ {cfg_name}  (train: {len(X_train)} nyata + {n_synth_used} sintetis"
              + (f" → {len(X_fit)} setelah augmentasi" if aug_copies else "") + ")")

        for model_name in model_names:
            model = build_candidates()[model_name]

            cv = None
            if args.cv:
                cv = cv_score(build_candidates()[model_name], X_train, y_train, g_train)

            t0 = time.time()
            model.fit(X_fit, y_fit)
            fit_seconds = round(time.time() - t0, 2)

            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            size_kb = serialize_size_kb(model_name, model, X_fit, y_fit, classes)

            rows.append({
                "config": cfg_name,
                "model": model_name,
                "n_train_real": len(X_train),
                "n_train_synth": n_synth_used,
                "n_fit": len(X_fit),
                "test_accuracy": round(acc, 4),
                "macro_f1": round(f1_macro, 4),
                "cv_accuracy": round(cv, 4) if cv is not None else None,
                "export_size_kb": size_kb,
                "fit_seconds": fit_seconds,
            })
            print(f"    {model_name:<4} acc={acc:.4f}  macroF1={f1_macro:.4f}  "
                  f"{'tambahan ' + str(size_kb) + ' KB  ' if size_kb else ''}({fit_seconds}s)")

    # --- Tabel + rekomendasi ---------------------------------------------------
    print("\n" + "=" * 78)
    header = f"{'Konfigurasi':<28} {'Model':<5} {'n_fit':>6} {'Acc (SI)':>9} {'MacroF1':>8} {'Δ vs base':>10}"
    print(header)
    print("-" * 78)

    baseline = {(r["model"]): r["test_accuracy"] for r in rows if r["config"] == "baseline"}
    best = {}
    for r in rows:
        delta = r["test_accuracy"] - baseline.get(r["model"], r["test_accuracy"])
        delta_str = "—" if r["config"] == "baseline" else f"{delta * 100:+.2f} poin"
        print(f"{r['config']:<28} {r['model']:<5} {r['n_fit']:>6} "
              f"{r['test_accuracy']:>9.4f} {r['macro_f1']:>8.4f} {delta_str:>10}")
        if r["config"] != "baseline":
            cur = best.get(r["model"])
            if cur is None or r["test_accuracy"] > cur["test_accuracy"]:
                best[r["model"]] = r
    print("=" * 78)

    recommendations = []
    for model_name in model_names:
        base_acc = baseline.get(model_name)
        if base_acc is None:
            continue
        top = best.get(model_name)
        if top is None:
            recommendations.append(f"{model_name.upper()}: hanya baseline yang diuji.")
            continue
        gain = top["test_accuracy"] - base_acc
        if gain < MARGINAL_GAIN:
            recommendations.append(
                f"{model_name.upper()}: TIDAK ada konfigurasi yang mengalahkan baseline secara berarti "
                f"(selisih terbaik {gain * 100:+.2f} poin). Pakai baseline — lebih sederhana & model lebih kecil."
            )
        else:
            recommendations.append(
                f"{model_name.upper()}: pakai '{top['config']}' "
                f"(+{gain * 100:.2f} poin, akurasi {top['test_accuracy'] * 100:.1f}%)."
            )

    print("\nRekomendasi:")
    for rec in recommendations:
        print(f"  • {rec}")

    # --- Simpan laporan --------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": stamp,
        "split": split_info,
        "n_real": int(len(X_real)),
        "n_synthetic_available": n_synth,
        "n_test": int(len(X_test)),
        "classes": classes,
        "configs": configs,
        "rows": rows,
        "recommendations": recommendations,
    }
    json_path = REPORTS_DIR / f"benchmark_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md = [
        "# Benchmark: baseline vs augmentasi vs data sintetis",
        "",
        f"- Tanggal: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        f"- Protokol split: **{split_info['protocol']}**",
        f"- Grup uji (tidak dilihat saat training): {', '.join(split_info['test_groups']) or '—'}",
        f"- Data: {len(X_real)} sampel nyata, {len(classes)} kelas"
        + (f", {n_synth} sampel sintetis tersedia" if has_synth else ""),
        f"- Test set: {len(X_test)} sampel nyata (identik untuk semua konfigurasi)",
        "",
        "| Konfigurasi | Model | Sampel train | Akurasi (SI) | Macro F1 | Δ vs baseline | Ukuran model |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        delta = r["test_accuracy"] - baseline.get(r["model"], r["test_accuracy"])
        delta_str = "—" if r["config"] == "baseline" else f"{delta * 100:+.2f} poin"
        size = f"{r['export_size_kb']} KB" if r["export_size_kb"] else "—"
        md.append(
            f"| {r['config']} | {r['model']} | {r['n_fit']} | {r['test_accuracy'] * 100:.2f}% | "
            f"{r['macro_f1'] * 100:.2f}% | {delta_str} | {size} |"
        )
    md += ["", "## Rekomendasi", ""]
    md += [f"- {rec}" for rec in recommendations]
    md += ["", "> Catatan: akurasi diukur pada data nyata yang belum pernah dilihat model "
               "(signer-independent). Data sintetis tidak pernah masuk test set.", ""]
    md_path = REPORTS_DIR / f"benchmark_{stamp}.md"
    md_path.write_text("\n".join(md))

    print(f"\nLaporan JSON    : {json_path}")
    print(f"Tabel markdown  : {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
