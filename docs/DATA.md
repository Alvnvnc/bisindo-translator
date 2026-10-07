# Panduan Persiapan Data — Jemari

> **Note (EN):** this is the team's internal working document, written in
> Bahasa Indonesia. The English-facing materials are `README.md`, the web UI,
> and `docs/devpost-submission.md`. CLI tooling output is also Bahasa Indonesia
> (team-facing); see README §Data format for the English summary.

> **Prinsip:** model bagus tidak menyelamatkan data buruk. Di project ini, kelas
> landmark sangat kecil (63 dimensi), jadi kualitas & keberagaman data jauh lebih
> menentukan daripada arsitektur model. Kerjakan dokumen ini dulu, baru training.

Dokumen ini mengikuti praktik standar: *Datasheets for Datasets* (Gebru et al.),
protokol evaluasi *signer-independent* yang dipakai komunitas Sign Language
Recognition (lihat WL-BISINDO di §3), dan disiplin anti-kebocoran data
(*data leakage*) pada pembagian train/test.

---

## 1. Tentukan spesifikasi SEBELUM mencari data

Mencari data tanpa spesifikasi = membuang waktu. Isi ini dulu:

| Keputusan | Pilihan Jemari | Alasan |
|---|---|---|
| Bahasa isyarat | **BISINDO** (bukan SIBI) | BISINDO adalah bahasa isyarat alami komunitas Tuli Indonesia; SIBI adalah sistem isyarat formal. Verifikasi klaim ini dengan sumber resmi/komunitas sebelum menuliskannya di submission. |
| Level isyarat | **Alfabet A–Z (statis)** untuk v1 | Pipeline kita single-frame. Isyarat kata butuh urutan gerakan (time series) → beda arsitektur. |
| Bentuk data | Landmark 63-d (21 titik × x,y,z) | Hasil normalisasi; invarian posisi & ukuran tangan. |
| Format file | CSV `class,f0..f62,session,signer` | Format seragam untuk rekaman sendiri maupun hasil ekstraksi dataset gambar. |
| Metadata wajib | `session` + `signer` per sampel | Tanpa ini, split signer-independent tak mungkin → akurasi bisa palsu. |
| Volume target | 300 sampel/kelas dari **≥ 3 periset** | Mulai 100; tambah sampai kurva validasi mendatar (lihat §5.1). |
| Lisensi data kita sendiri | CC BY 4.0 | Terbuka untuk riset lanjutan, sesuai semangat challenge. |

Catatan penting: **BISINDO punya varian regional** (DKI Jakarta, Banten, dst.).
Tentukan satu varian acuan untuk v1 dan tulis di dokumentasi — jangan campur
varian tanpa catatan, karena bentuk isyaratnya bisa berbeda.

---

## 2. Kriteria "data yang cocok"

Sebuah dataset layak dipakai jika **semua** poin ini terpenuhi:

1. **Kelasnya cocok** dengan spesifikasi (alfabet A–Z BISINDO).
2. **Lisensinya jelas** dan mengizinkan penggunaan riset + turunan (CC BY, MIT, dsb.).
3. **Formatnya bisa dikonversi** ke pipeline kita (gambar/video → MediaPipe; atau
   sudah berupa keypoint/landmark).
4. **Ada keragaman signer** (atau minimal, itu dikenali sebagai keterbatasan).
5. **Ada dokumentasi** asal-usul: siapa yang tampil, kondisi perekaman, izin (consent).

Aturan berhenti: **lisensi tidak jelas = jangan dipakai.** Titik. Jangan juga
memakai dataset lalu menulis "dipakai dari internet" di submission — juri teknis
akan menanyakannya.

---

## 3. Katalog sumber data (sudah diverifikasi)

### BISINDO — alfabet (paling relevan)

| Sumber | Isi | Lisensi | Catatan |
|---|---|---|---|
| [Mendeley 10.17632/ywnjpbcz8m.1](https://data.mendeley.com/datasets/ywnjpbcz8m) | Gambar A–Z; perangkat, latar, dan individu beragam; versi original, terkompresi, background-removed, biner | **CC BY 4.0** | Universitas Multimedia Nusantara; disebutkan ada consent & ethical approval. Kandidat utama. |
| [Mendeley 10.17632/235c78xbmk.2](https://data.mendeley.com/datasets/235c78xbmk) | Video + frame; 19.760 sampel, 26 kelas; indoor (terang/redup) & outdoor | **CC BY 4.0** | Versi publik hanya 1 aktor (wajah diblur) — keragaman signer terbatas, bagus untuk variasi pencahayaan. |
| [Mendeley 4xnkvr88tk](https://data.mendeley.com/datasets/4xnkvr88tk) | Gambar 26 huruf A–Z | cek di halaman | Belum diverifikasi detail — periksa sendiri. |
| [GitHub rhiosutoyo/…-Dataset](https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset) | 520 foto (20/huruf), anotasi bounding box Pascal VOC | **MIT** | Terkait paper IEEE ICRAIE 2023. Ukuran kecil; bagus untuk validasi tambahan. |
| [Kaggle agungmrf/indonesian-sign-language-bisindo](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo) | Alfabet BISINDO | **Cek lisensi di halaman** | Jangan unduh sebelum lisensi diverifikasi. |
| [Roboflow: cv-yjopp/sign-language-bisindo](https://universe.roboflow.com/cv-yjopp/sign-language-bisindo) | ±826 gambar + model | Cek per versi di Roboflow | Roboflow menandai lisensi per dataset; sebagian hanya subset gratis. |

### SIBI (hati-hati: bahasa isyarat berbeda)

| Sumber | Isi | Lisensi | Catatan |
|---|---|---|---|
| [Kaggle mlanangafkaar SIBI Alphabets](https://www.kaggle.com/datasets/mlanangafkaar/datasets-lemlitbang-sibi-alphabets) | 520 (RAW) + 1.312 (v02) foto alfabet, terverifikasi LEMLITBANG SIBI | **CC BY 4.0** | Mutu bagus, tapi **SIBI ≠ BISINDO** — jangan dicampur tanpa alasan ilmiah. |
| [Kaggle SIBI Keypoints (20 kelas)](https://www.kaggle.com/datasets/dersterngreifer/sibi-sign-language-keypoints-dataset-20-classes) | 270.038 file keypoint, ±326 MB | Cek halaman | Kalau formatnya landmark, bisa **langsung** masuk pipeline; verifikasi dulu. |
| [Kaggle gesturesibi](https://www.kaggle.com/datasets/andresaputraginting/gesturesibi) | 26 file CSV, ±2 MB | Cek halaman | Kemungkinan sudah berupa landmark — paling murah untuk diuji. |

### Untuk level kata (stretch, bukan v1)

| Sumber | Isi | Lisensi | Catatan |
|---|---|---|---|
| [WL-BISINDO](https://github.com/AceKinnn/WL-BISINDO) · [Kaggle](https://www.kaggle.com/datasets/glennleonali/wl-bisindo) | 1.600 video, 32 isyarat kata, 5 periset, varian Banten, ±2,16 GB | **CC BY-NC 4.0** (non-komersial!) | Berisi protokol split signer-dependent & signer-independent — contoh terbaik disiplin split. Bukti empiris: SD 97% vs SI turun sampai 48% (§6). |

### Di mana lagi mencari (resep pencarian)

- **Mendeley Data / Zenodo** — cari `"BISINDO" dataset`, `"bahasa isyarat" dataset`.
  Kumpulan data akademik; lisensi biasanya tercantum jelas (CC BY).
- **Kaggle / Hugging Face Datasets** — cari `BISINDO`, `SIBI`, `Indonesian sign language`.
- **Roboflow Universe** — cari `bisindo`; perhatikan batas versi gratis.
- **GitHub** — topik [`bisindo`](https://github.com/topics/bisindo),
  [`indonesian-sign-language`](https://github.com/topics/indonesian-sign-language); repo paper biasanya menautkan dataset aslinya.
- **Paper + DOI** — cari di Google Scholar `BISINDO dataset recognition`; paper yang
  baik menyebut DOI dataset (biasanya Mendeley/Zenodo). Ikuti dari paper, bukan dari
  situs agregator tak jelas.
- **Komunitas** — organisasi Tuli (mis. Gerkatin) & kampus. Untuk proyek ini,
  kolaborasi langsung dengan periset BISINDO lebih bernilai daripada dataset besar
  tapi tanpa izin.

---

## 4. Kartu evaluasi dataset (isi untuk setiap kandidat)

Sebelum memakai dataset apa pun, jawab 10 pertanyaan ini (simpan jawabannya —
itu menjadi bagian dokumentasi submission):

1. Nama & DOI/URL dataset?
2. Lisensi persisnya apa? (baca halaman lisensinya, bukan lisensi repo)
3. Siapa periset/aktornya? Berapa banyak? Tangan kanan/kiri?
4. Kondisi perekaman apa saja (latar, cahaya, jarak, perangkat)?
5. Berapa sampel per kelas? Seimbangkah?
6. Formatnya apa? (gambar/video/keypoint). Perlu langkah konversi apa?
7. Ada izin/consent? Bagaimana privasi subjek dilindungi?
8. Keterbatasan yang diakui penulisnya?
9. Apakah perlu sitasi? (tulis di `ATTRIBUTIONS.md`)
10. Apakah jumlah & keragamannya cukup untuk menguji generalisasi antar-individu?

---

## 5. Membuat data sendiri (protokol resmi proyek ini)

Dataset publik BISINDO bervariasi lisensinya dan umumnya sedikit periset. **Rekaman
sendiri adalah sumber utama** — lebih orisinal di mata juri, dan metadata-nya
lengkap.

### 5.1 Berapa banyak? Ikuti kurva, jangan tebak

- Mulai dari **100 sampel/kelas** (cukup untuk KNN/MLP dengan 63 fitur).
- Latih, catat akurasi test **signer-independent** (§6), tambah data, ulangi.
- Berhenti menambah saat kurva akurasi **mendatar** (kenaikan < 1–2%).
  Untuk landmark, biasanya plateau di sekitar **300/kelas** — di situlah target kita.
- Ingat: 1.000 sampel dari 1 orang < 300 sampel dari 3 orang. **Keragaman periset
  mengalahkan jumlah.**

### 5.2 Sumbu variabilitas (wajib bervariasi saat merekam)

| Sumbu | Variasi yang harus ada | Kenapa |
|---|---|---|
| Periset | ≥ 3 orang (beda ukuran tangan & warna kulit) | Generalisasi antar-individu |
| Tangan | Tentukan: tangan kanan saja, atau kiri juga (catat) | Sisi tangan mengubah geometri |
| Jarak | 30–80 cm dari kamera | Variasi skala nyata (normalisasi punya batas) |
| Sudut | hadap depan, sedikit miring kiri/kanan/atas | Orientasi tangan di dunia nyata |
| Pencahayaan | terang, redup, dari samping | MediaPipe mendeteksi buruk saat kontras ekstrem |
| Latar | polos & berantakan | Kondisi loket sebenarnya |
| Gerakan | tenangkan pose, geser sedikit saat merekam | Menghindari frame kembar |

### 5.3 Sesi perekaman

- Satu kelas direkam dalam **≥ 2 sesi berbeda** (ideal: hari berbeda).
  Variasi antar-waktu itu penting dan tidak bisa ditipu.
- Satu kali tekan "Rekam" = satu sesi (ID otomatis: `ses-YYYYMMDD-hhmmss-xxxx`).
- Isi **ID periset** (mis. `S01`) di alat rekam — tanpa itu, audit tidak bisa
  memverifikasi cakupan signer.

### 5.4 Konvensi labeling (tulis & patuhi)

- **Satu frame = satu label.** Frame transisi antar-isyarat jangan disimpan.
- Tulis **definisi eksplisit** tiap huruf yang dipakai tim (bentuk jari, orientasi
  telapak). Kalau dua orang di tim berbeda pendapat soal bentuk huruf, model akan
  menerima label yang bertentangan — itu sumber error terbesar.
- Tangan tidak terdeteksi → sampel tidak tersimpan (sudah otomatis di alat rekam).
- Keraguan → lebih baik **tidak** menyimpan sampel daripada menyimpan label ragu.

### 5.5 Alur kerja rekam (perintah)

```bash
source .venv/bin/activate
python -m http.server 8000
# buka http://localhost:8000/web/capture.html
```

1. Isi **ID periset** (mis. `S01`), pilih kelas (mis. `A`).
2. Tekan **Rekam**; gerakkan tangan perlahan selama perekaman (jangan kaku).
3. Ganti kelas → ulangi. Setelah 1 putaran penuh, **ekspor CSV** dan simpan ke
   `data/samples/` (nama file: `manual_S01_20261005.csv`).
4. Rekaman orang berikutnya: ganti ID periset. Ekspor file terpisah per periset.
5. Ulangi di sesi hari lain, ekspor lagi.

> Data tersimpan di localStorage browser — **ekspor rutin** atau data hilang saat
> browser dibersihkan.

### 5.6 Untuk dataset gambar publik (konversi)

1. Unduh dataset, simpan ke `data/raw/<KELAS>/…` (subfolder **per periset** bila ada:
   `data/raw/A/S01/…` → kolom `signer` terisi otomatis).
2. Jalankan `python scripts/extract_landmarks.py --signer dataset_id` (atau tanpa
   `--signer` bila subfolder periset sudah disiapkan).
3. Cek laporannya: berapa gambar yang tangannya **tidak** terdeteksi. Kalau > 20%
   terlewat, kualitas dataset itu perlu dipertanyakan.
4. Update `ATTRIBUTIONS.md` dengan DOI, lisensi, dan sitasi.

---

## 6. Protokol split — bagian paling sering dilanggar

**Masalah:** kalau frame dari orang yang sama masuk ke train DAN test, model hanya
menghafal orangnya, bukan isyaratnya. Akurasi naik, kegunaan nyata nol.
Inilah bedanya **SD (signer-dependent)** dan **SI (signer-independent)**.

Bukti dari dataset WL-BISINDO (BISINDO level kata, 5 periset):

| Pengaturan | Akurasi model terbaik |
|---|---|
| Signer-**dependent** (train & test orang sama) | 93,75% – 97,08% |
| Signer-**independent** (test = orang yang tidak dilihat model) | 48,13% – 90,95% |

Artinya: evaluasi SI adalah satu-satunya angka yang jujur untuk sistem yang akan
dipakai orang baru.

**Protokol kita** (diimplementasikan di `scripts/train.py`):

- Default: `--group-column signer` → seluruh sampel satu periset hanya ada di satu
  sisi (train **atau** test) — memakai `StratifiedGroupKFold`.
- Kalau kolom signer tidak ada/kurang dari 2 nilai, script akan menyatakan bahwa
  splitnya acak dan akurasi **terlalu optimistis** (jangan pakai angka itu di poster).
- **Bekukan test set**: begitu mulai tuning model, jangan pernah mengubah split.
- **Augmentasi hanya untuk train** — jangan pernah augmentasi data test.

```bash
# Split default: signer-independent
python scripts/train.py

# Membandingkan (untuk laporan): split acak
python scripts/train.py --group-column none

# Split per sesi rekaman (lebih ketat: menguji generalisasi antar-sesi)
python scripts/train.py --group-column session
```

---

## 7. Teori augmentasi (ruang landmark)

Karena fitur kita sudah berupa landmark ternormalisasi, aturan augmentasinya
berbeda dari augmentasi gambar:

| Transformasi | Boleh? | Alasan |
|---|---|---|
| Translasi (geser) | ❌ | Normalisasi sudah menghilangkannya |
| Skala (zoom) | ❌ | Normalisasi sudah menghilangkannya |
| Rotasi kecil (± 10–15°) | ✅ | Meniru variasi sudut tangan; aman untuk huruf |
| Noise Gaussian kecil (σ ≈ 0.01) | ✅ | Meniru jitter deteksi MediaPipe |
| Flip horizontal | ❌ | Mengubah makna huruf & sisi tangan — label jadi salah |
| Warp besar / distorsi | ❌ | Tidak realistis; mendorong model belajar pola palsu |

Aturan emas: augmentasi **hanya pada fold training**, diterapkan setelah split,
dan diverifikasi tidak bocor ke test (bandingkan hash/indeks).

---

## 7b. Data sintetis — metode komunitas & aturan pakai

**Prinsip dasar dari literatur:** sintetis hanya membantu bila **real-anchored**
(dijalankan dari data nyata, bukan dari nol). Tanpa jangkar nyata, model belajar
halusinasi generatornya. (arXiv 2605.09699 — *When Does Synthetic Pose Data Help?*)

### Metode yang dipakai komunitas (dari termurah ke termahal)

| # | Metode | Cara kerja | Catatan untuk kita |
|---|---|---|---|
| 1 | Augmentasi prosedural | Rotasi/noise di ruang landmark | Sudah ada (`--augment`); tidak menambah mode gerakan baru |
| 2 | Interpolasi dalam kelas (MixUp) | `λ·x_a + (1−λ)·x_b`, `x_a,x_b` satu kelas & berdekatan | Aman & murah; batasi ke tetangga terdekat agar tidak menjembatani dua gaya gerakan |
| 3 | Model generatif statistik | PCA per kelas + sampling Gaussian di ruang komponen | Yang kita implementasikan (`generate_synthetic.py`); ringan, tanpa GPU |
| 4 | GAN / VAE / Diffusion pada skeleton | Generator dilatih pada data nyata (mis. *Skeleton-Based Data Augmentation using Adversarial Learning*; pose GAN di Frontiers 2024) | Butuh data & waktu jauh lebih banyak; risiko mode collapse. Tidak untuk sprint 5 hari |
| 5 | *Sign stitching* / SLP-based | Menyambung segmen isyarat untuk membuat sampel baru (arXiv 2506.09643, 2508.14345) | Untuk isyarat berurutan (kata/kalimat). Pretraining sintetis bisa melampaui augmentasi biasa — relevan untuk v2 |
| 6 | Video generatif (mis. Firefly) | Menghasilkan video isyarat dari prompt | Studi MDPI 2025: sulit mendapatkan isyarat yang benar via prompt — **tidak dapat diandalkan untuk akurasi**, hanya untuk ilustrasi |

### Yang justru TIDAK disarankan: SMOTE

Konsensus komunitas untuk data tabular/fitur (seperti landmark 63-d kita):
SMOTE tidak membantu classifier modern, **merusak kalibrasi probabilitas**, dan
lemah pada data kecil / kelas yang kompleks (diskusi r/MachineLearning; C. Molnar
"Don't fix your imbalanced data"; review PMC10789107). Karena itu kita **tidak
memakai SMOTE** — kita pakai metode 1–3 di atas.

### Aturan pakai di proyek ini (ditegakkan oleh kode)

1. **Sintetis tidak pernah masuk test.** `scripts/generate_synthetic.py` menandai
   semua sampel dengan `signer='synthetic'`; `scripts/train.py` hanya menerimanya
   ke train via `--include-synthetic`, dan test selalu 100% data nyata.
2. **CV dilaporkan pada data nyata saja** — angka sintetis tidak pernah jadi bukti.
3. **Batas dilusi** — maksimum 2× jumlah nyata per kelas (`--max-fraction`);
   kelas dengan < 20 data nyata dilewati.
4. **Kontrol kualitas otomatis** — generator menolak dipakai bila label-match < 95%
   atau rasio jarak sintetis/nyata > 1.5.
5. **Bukan pengganti data nyata** — kalau akurasi signer-independent tidak naik,
   matikan sintetis. Jangan pernah melaporkan metrik dari test sintetis.

### Cara pakai

```bash
# 1. Bandingkan SEMUA konfigurasi sekaligus pada split yang sama —
#    hasilnya berupa tabel + rekomendasi otomatis (jujur, berbasis angka).
python scripts/benchmark.py

# 2. Latih dengan konfigurasi yang menang (contoh: bila 'sintetis' terbukti unggul)
python scripts/train.py --include-synthetic

# Periksa sintetis tanpa menulis file
python scripts/generate_synthetic.py --dry-run

# Bandingkan manual: baseline vs sintetis
python scripts/train.py
python scripts/train.py --include-synthetic
```

Tabel benchmark tersimpan di `reports/benchmark_*.md` — lampirkan di submission
sebagai bukti proses (sumbangan untuk penilaian Presentasi & Dokumentasi).

---

## 8. QC otomatis: `scripts/audit_dataset.py`

Jalankan SEBELUM setiap training serius:

```bash
python scripts/audit_dataset.py
```

Yang diperiksa (beserta cara membaca hasilnya):

1. **Distribusi kelas** — kelas < 100 sampel → tambah data. Rasio timpang > 3× →
   fokuskan rekaman pada kelas yang sedikit (jangan buang data kelas besar).
2. **Cakupan metadata** — berapa periset/sesi, kelas mana yang < 2 periset.
   Target: ≥ 3 periset; kelas dengan 1 periset tidak bisa diuji SI.
3. **Kewarasan fitur** — NaN/Inf, rentang nilai, pergelangan tidak di origin
   (tanda data diproses tanpa normalisasi yang benar).
4. **Duplikat** — pasangan nyaris identik dalam kelas yang sama. Duplikat membuat
   test terlihat bagus palsu; hapus.
5. **Outlier** — kandidat salah label (jarak ke centroid kelasnya ekstrem).
   Periksa 10 teratas satu per satu: perbaiki label atau hapus.
6. **Risiko tertukar** — pasangan kelas dengan centroid berdekatan. Tindaknya:
   perjelas definisi isyarat & tambah variasi, bukan menambah model yang rumit.

Laporan JSON tersimpan di `reports/data_audit_*.json` — lampirkan ringkasannya di
submission sebagai bukti proses (bagian dari nilai Presentasi & Dokumentasi).

---

## 9. Dokumentasi & etika (bagian dari *Datasheets for Datasets*)

| Aspek | Aturan proyek ini |
|---|---|
| Consent | Setiap periset/orang yang tampil memberi izin sadar untuk datanya dipakai di proyek ini & dilisensikan CC BY 4.0. Untuk anak di bawah umur: izin orang tua/wali. |
| Privasi | Pipeline hanya menyimpan **63 angka landmark**, bukan gambar/video. Wajah tidak pernah tersimpan. Tuliskan ini sebagai nilai jual privasi. |
| Komunitas | Untuk klaim dampak ke komunitas Tuli, libatkan penutur BISINDO (mis. komunitas/relawan) — idealnya mereka ikut merancang, bukan jadi objek. Jangan klaim "dibuat untuk komunitas Tuli" tanpa pernah berinteraksi dengan komunitasnya. |
| Kompensasi | Bila merekam orang di luar tim dalam sesi panjang, pertimbangkan kompensasi wajar. |
| Sitasi | Setiap dataset pihak ketiga & repo yang dipakai → `ATTRIBUTIONS.md` + seksi Technologies Used di Devpost. |
| Klaim | Dilarang menulis angka yang tidak bisa ditunjukkan buktinya. |

---

## 10. Versioning data

- Simpan snapshot training yang dipakai untuk angka final, mis. beri nama file
  `data/samples/v1_20261007_*.csv` — jangan menimpa file lama.
- Catat di `reports/`: versi data + hasil metrik. Satu angka akurasi → satu versi data.
- **Test set dibekukan**: setelah ditentukan, tidak boleh diedit/ditambah.
- Kalau data diperbaiki (label salah dihapus), itu menjadi versi baru + split baru,
  dan semua angka lama ditandai kedaluwarsa.

---

## 11. Rencana eksekusi data (2 hari, realistis)

**Hari 1 — fondasi**
- [ ] Baca dokumen ini; putuskan BISINDO vs SIBI (default rekomendasi: BISINDO) dan v1 = alfabet.
- [ ] Tulis definisi 26 huruf yang dipakai tim (1–2 baris/huruf).
- [ ] Uji `capture.html`: rekam 2 kelas × 1 sesi; ekspor CSV; jalankan
      `audit_dataset.py` → pastikan alurnya nyaman.
- [ ] Unduh 1 dataset publik kandidat (§3), verifikasi lisensinya, ekstrak dengan
      `extract_landmarks.py`. Catat berapa % gambar gagal terdeteksi.

**Hari 2 — produksi data**
- [ ] 3 periset × 26 huruf × 2 sesi. Target: ≥ 150 sampel/kelas dulu (naikkan ke
      300 bila waktu memungkinkan).
- [ ] Ekspor semua CSV ke `data/samples/`, jalankan audit → perbaiki temuan
      (duplikat, outlier, kelas timpang).
- [ ] `python scripts/train.py` → catat akurasi **SI** di laporan.
- [ ] Tulis ringkasan data untuk submission: jumlah sampel, periset, sesi,
      sumber dataset + lisensi, temuan audit, keterbatasan.

---

## 12. Referensi teori

- Gebru et al. — *Datasheets for Datasets* (dokumentasi dataset).
- Kindy, Leonali & Lucky (2025) — *Word-Level BISINDO* (DOI 10.1016/j.procs.2025.08.277):
  protokol SD/SI dan hasil empirisnya.
- scikit-learn — [Cross-validation dengan grup](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
  (`StratifiedGroupKFold`, `GroupShuffleSplit`).
- Praktik *data-centric AI*: perbaiki data dulu sebelum menambah kompleksitas model.
