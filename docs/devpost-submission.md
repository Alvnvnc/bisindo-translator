# Draft Submission Devpost — HandTalk ID

Template ini mengikuti persis field yang diminta panitia. Isi bagian `[...]`,
**verifikasi setiap angka dari sumber resmi** sebelum menyalin ke Devpost.

---

## Project Title

**HandTalk ID** — jembatan bahasa isyarat di layanan publik

*(Alternatif bila nama sudah dipakai peserta lain: "HandTalk ID: BISINDO Bridge",
"IsyaratKita", "Jari Bicara")*

---

## Inspired by

ML Empowerment Build Challenge 3.0 — kategori bebas (tidak ada tema wajib).

---

## What it does

HandTalk ID menerjemahkan isyarat tangan Bahasa Isyarat Indonesia menjadi teks dan
suara secara real-time, langsung di browser. Pengguna cukup menyalakan kamera dan
memberi isyarat; huruf demi huruf muncul dan bisa langsung diucapkan dengan suara
bahasa Indonesia agar lawan bicara yang tidak memahami bahasa isyarat tetap bisa
mengikuti percakapan.

Seluruh pemrosesan berjalan di perangkat pengguna — video tidak pernah dikirim ke
server, sehingga privasi terjaga dan aplikasi tetap ringan untuk perangkat kelas
menengah.

---

## Problem Statement

Di loket layanan publik (puskesmas, kantor desa, bank, sekolah), komunikasi antara
pengguna bahasa isyarat dan petugas sering putus karena tidak ada penerjemah
tersedia. Menulis di kertas memang jadi jalan keluar, tetapi lambat, menyulitkan
kedua pihak, dan tidak setara.

Juru bahasa isyarat bersertifikat jumlahnya terbatas dan tidak selalu tersedia saat
dibutuhkan, terutama di daerah. `[VERIFIKASI & SITIR: jumlah populasi Tuli di
Indonesia dan rasio ketersediaan juru bahasa isyarat — cari sumber resmi, mis.
data Kementerian Sosial / Gerkatin / WHO, lalu tulis angkanya di sini.]`

Akibatnya, kebutuhan sederhana seperti menjelaskan gejala ke dokter atau mengurus
dokumen menjadi pengalaman yang melelahkan dan bergantung pada orang lain.

---

## Solution Overview

HandTalk ID mengubah kamera ponsel atau laptop menjadi penerjemah isyarat yang
selalu tersedia, tanpa aplikasi khusus dan tanpa biaya langganan:

1. **Deteksi tangan** — MediaPipe Hand Landmarker mengekstrak 21 titik kunci tangan
   per frame, berjalan di WebAssembly sehingga tidak butuh GPU khusus.
2. **Normalisasi** — landmark digeser ke titik pergelangan dan diskalakan terhadap
   ukuran tangan, sehingga model tidak bergantung pada posisi tangan di frame
   maupun perbedaan ukuran tangan (anak-anak vs dewasa).
3. **Klasifikasi** — vektor 63 dimensi dimasukkan ke classifier yang **kami latih
   sendiri** (KNN / MLP kecil), lalu hasil beberapa frame dirata-ratakan secara
   temporal agar teks tidak berkedip antar huruf.
4. **Output dua arah** — teks ditampilkan besar dan dapat diucapkan lewat Web
   Speech API dalam bahasa Indonesia.

Karena seluruh pipeline berjalan di klien, aplikasi bisa dipakai di ruang tunggu
dengan koneksi terbatas tanpa mengirim data pribadi siapa pun.

---

## Key Features

- **Terjemahan isyarat real-time** dari webcam, tanpa instalasi
- **Suara bahasa Indonesia** (Web Speech API) agar lawan bicara langsung paham
- **Transkrip kalimat** yang bisa disalin atau dibersihkan
- **Indikator keyakinan** per huruf — pengguna tahu kapan sistem ragu
- **100% on-device** — video tidak pernah meninggalkan perangkat
- **Mode Latih** — siapa pun bisa menambah sampel isyaratnya sendiri lewat
  `capture.html`, lalu melatih ulang model
- **Smoothing temporal** — voting beberapa frame + ambang keyakinan untuk hasil stabil

---

## Technologies Used

| Teknologi | Peran |
|---|---|
| MediaPipe Tasks Vision (Hand Landmarker) | Deteksi 21 landmark tangan di browser (WASM/WebGL) |
| JavaScript (ES modules, Canvas API) | Runtime inference, UI, visualisasi landmark |
| Web Speech API | Text-to-speech bahasa Indonesia |
| Python 3.12 + scikit-learn | Training & evaluasi classifier (KNN, MLP, RandomForest) |
| NumPy / OpenCV | Praproses data, ekstraksi landmark dari dataset gambar |
| localStorage | Penyimpanan dataset rekaman di sisi klien |
| `[opsional] Featherless.ai` | `[isi bila dipakai, mis. untuk parafrase kalimat isyarat]` |
| `[opsional] Mobbin` | Referensi pola UI saat merancang tampilan |

Kredit lengkap komponen open-source dan lisensinya ada di
[`ATTRIBUTIONS.md`](../ATTRIBUTIONS.md).

---

## Model & Evaluasi

Isi setelah training selesai — **pakai angka nyata dari `reports/`**, jangan dikarang:

- Dataset: `[N]` sampel, `[N]` kelas, sumber: `[rekaman sendiri / nama dataset + lisensi]`
- Protokol split: signer-independent (grup uji: `[Sxx]`) — bukan split acak
- Akurasi test: `[X]%` (hold-out `[Y]%`), macro-F1: `[X]%`
- Model terpilih: `[KNN k=… / MLP …]` berdasarkan cross-validation
- Perbandingan konfigurasi (dari `reports/benchmark_*.md`): `[tempel tabelnya]`
- Kelas paling sering tertukar: `[mis. M ↔ N]` dan langkah perbaikannya
- Laporan audit data: `[ringkas hasil scripts/audit_dataset.py]`

---

## Target Users

1. **Pengguna bahasa isyarat (BISINDO/SIBI)** — terutama yang sering berurusan dengan
   layanan publik tanpa pendamping.
2. **Petugas layanan publik** — puskesmas, kantor desa, bank, sekolah, yang perlu
   berkomunikasi tanpa juru bahasa isyarat.
3. **Keluarga & relawan** — yang ingin berkomunikasi atau belajar isyarat dasar
   lewat Mode Latih.

---

## Social Impact Statement (opsional tapi dianjurkan)

`[Tulis 3–4 kalimat: masalah kesenjangan akses layanan, siapa yang terbantu, dan
bagaimana proyek ini bisa dipakai gratis tanpa biaya infrastruktur. Sertakan angka
yang sudah diverifikasi + sumbernya.]`

---

## Project Files (wajib minimal 1)

- **Video demo 2–3 menit** — struktur:
  1. 0:00–0:15 — masalah (satu kalimat + tunjukkan situasi loket)
  2. 0:15–0:45 — demo langsung: isyarat → huruf → kalimat → suara
  3. 0:45–1:30 — cara kerja (landmark overlay, pipeline on-device)
  4. 1:30–2:15 — Mode Latih + hasil evaluasi model (tampilkan angka akurasi)
  5. 2:15–2:45 — dampak & langkah berikutnya
- **Screenshot 3–5 layar:** demo utama, overlay landmark, Mode Latih/capture,
  tabel dataset, laporan metrik
- Rekam video dengan resolusi ≥ 720p, tangan terlihat jelas, dan **tampilkan teks
  hasil terjemahan** di layar (juri mungkin menonton tanpa suara)

## Project Link / Repository

- Repo GitHub (public): `[https://github.com/…]`
- Live demo: `[GitHub Pages / Vercel / Netlify — perlu HTTPS agar kamera berfungsi]`
- Catatan: `model.json` hasil training harus ikut ter-deploy

## Team Details

| Nama | Peran |
|---|---|
| `[…]` | `[mis. ML — dataset, training, evaluasi]` |
| `[…]` | `[mis. Frontend — UI, integrasi MediaPipe]` |
| `[…]` | `[mis. Riset pengguna, dokumentasi, video]` |

*(Solo juga diperbolehkan — cukup satu baris.)*

---

## Checklist sebelum submit

- [ ] Semua field di atas terisi, tidak ada `[placeholder]` tersisa
- [ ] Setiap angka punya sumber yang bisa diverifikasi
- [ ] Video & screenshot terunggah dan bisa diputar
- [ ] Link repo aktif dan public (bukan private)
- [ ] Live demo bisa dibuka di HP orang lain (HTTPS, kamera jalan)
- [ ] `ATTRIBUTIONS.md` mencantumkan semua komponen pihak ketiga
- [ ] Anggota tim terdaftar di submission
- [ ] Submit jauh sebelum deadline (9 Okt 11:45pm PDT)
