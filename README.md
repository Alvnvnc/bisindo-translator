# HandTalk ID

**Jembatan komunikasi Bahasa Isyarat Indonesia (BISINDO) di layanan publik.**

Aplikasi web yang menerjemahkan isyarat tangan menjadi teks dan suara secara *real-time*
langsung di browser — tanpa server, tanpa kamera cloud, dan tetap berfungsi untuk
pengguna dengan koneksi terbatas. Ditujukan untuk interaksi di loket layanan publik
(puskesmas, kantor desa, bank) antara pengguna BISINDO dan petugas yang tidak
menguasai bahasa isyarat, plus mode latihan untuk orang yang baru belajar.

- **Deadline submission:** 9 Oktober 2026, 11:45pm PDT (≈ 10 Okt 13:45 WIB)
- **Kompetisi:** [ML Empowerment Build Challenge 3.0](https://ml-build-challenge-3.devpost.com/)

---

## Arsitektur

```
Kamera browser
      │
      ▼
MediaPipe Hand Landmarker (WebAssembly, jalan di device)
      │  21 titik landmark per tangan (x, y, z)
      ▼
Normalisasi (translasi ke pergelangan + skala ukuran tangan)
      │  vektor fitur 63 dimensi
      ▼
Classifier hasil training (KNN / MLP kecil) ── dimuat dari web/model.json
      │
      ▼
Smoothing (voting N frame + ambang confidence)
      │
      ▼
Teks di layar  +  suara (Web Speech API, id-ID)
```

Seluruh inference berjalan **di sisi klien**. Data kamera tidak pernah meninggalkan
perangkat pengguna — ini bagian dari nilai jual proyek (privasi by design).

Alur kerja pengembangan:

1. **Kumpulkan data** — `web/capture.html` (rekam sampel landmark sendiri lewat webcam),
   atau `scripts/extract_landmarks.py` (konversi dataset gambar ke format yang sama).
2. **Latih model** — `scripts/train.py` → evaluasi + ekspor `web/model.json`.
3. **Jalankan demo** — buka `web/index.html`.

---

## Setup

Prasyarat: Python 3.12 (via `uv`), Node.js hanya untuk local server opsional.

```bash
# 1. Buat virtual environment Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. Install dependency
uv pip install -r requirements.txt

# 3. Uji pipeline dengan data sintetis (sekali saja, untuk memastikan alurnya jalan)
python scripts/make_sample_dataset.py
python scripts/train.py

# 4. Verifikasi Python ↔ JavaScript menghasilkan angka yang sama
python scripts/check_parity.py

# 5. Jalankan web demo
python -m http.server 8000
# buka http://localhost:8000/web/index.html       → demo terjemahan
# buka http://localhost:8000/web/capture.html     → rekam dataset sendiri
```

> Catatan: `http.server` diperlukan karena MediaPipe Tasks Vision memuat model lewat
> fetch, yang diblokir pada protokol `file://`. Untuk demo publik, deploy ke
> GitHub Pages / Vercel / Netlify (wajib HTTPS agar kamera diizinkan browser).

---

## Struktur repo

```
├── README.md
├── ATTRIBUTIONS.md            # kredit repo/dataset pihak ketiga + lisensi
├── LICENSE                    # MIT
├── requirements.txt
├── data/
│   ├── raw/                   # gambar/video mentah per kelas (opsional)
│   └── samples/               # CSV landmark: class,f0..f62  ← input training
├── scripts/
│   ├── extract_landmarks.py   # gambar/video → CSV landmark (MediaPipe Python)
│   ├── train.py               # latih + evaluasi + ekspor web/model.json
│   ├── make_sample_dataset.py # data sintetis untuk smoke test pipeline
│   ├── check_parity.py        # uji Python ↔ JS (jalankan tiap ubah normalisasi/model)
│   └── check_parity.mjs
├── web/
│   ├── index.html             # demo utama: terjemah real-time
│   ├── app.js                 # wiring kamera → klasifikasi → UI + TTS
│   ├── capture.html           # alat rekam dataset lewat webcam
│   ├── capture.js
│   ├── landmarks.js           # normalisasi + inference (inti matematika)
│   ├── hands.js               # wrapper MediaPipe Tasks Vision
│   ├── model.json             # hasil training (ikut di-commit agar demo bisa di-deploy)
│   └── style.css
├── reports/                   # metrik evaluasi tiap run training (tidak masuk git)
└── docs/
    ├── PLAN.md                # sprint 5 hari + aturan de-risking
    └── devpost-submission.md  # template submission Devpost
```

---

## Format data

Semua sumber data bermuara ke satu format CSV di `data/samples/`:

```
class,f0,f1,f2,...,f62
A,0.123,-0.456,...
```

- `class` — label isyarat (mis. huruf alfabet `A`–`Z`)
- `f0..f62` — 21 landmark × 3 koordinat (x, y, z), **sudah dinormalisasi**

Normalisasi (wajib identik antara Python dan JS — lihat `scripts/train.py` dan
`web/app.js`):

1. Geser semua titik sehingga pergelangan tangan (landmark 0) berada di origin.
2. Bagi dengan jarak terjauh dari pergelangan ke landmark mana pun (scale-invariant).
3. Ratakan menjadi vektor 63 dimensi.

Ini membuat model tidak bergantung pada posisi tangan di frame maupun ukuran tangan
pengguna (anak-anak vs dewasa).

---

## Status

- [x] Scaffold pipeline: landmark → training → ekspor → inference di browser
- [x] Alat rekam dataset (`web/capture.html`) + ekspor/impor CSV
- [x] Uji paritas Python ↔ JavaScript (`scripts/check_parity.py`) — lulus
- [x] Demo UI dasar: huruf besar, indikator keyakinan, transkrip, TTS `id-ID`
- [ ] Dataset huruf alfabet BISINDO (target: ≥ 300 sampel/kelas)
- [ ] Akurasi ≥ 85% pada test split dengan data nyata
- [ ] Uji dengan pengguna asli + video demo
- [ ] Mode latihan & frasa layanan (stretch)
- [ ] Deploy live demo + submission Devpost

> ⚠️ `web/model.json` dan `data/samples/synthetic.csv` saat ini berasal dari **data
> sintetis** hasil smoke test — bukan isyarat asli. Jalankan `scripts/train.py` lagi
> setelah dataset nyata terkumpul.

Rencana harian detail ada di [`docs/PLAN.md`](docs/PLAN.md), dan template submission
di [`docs/devpost-submission.md`](docs/devpost-submission.md).
