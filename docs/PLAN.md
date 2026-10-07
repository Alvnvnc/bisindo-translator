# Rencana Sprint 5 Hari — Sasmita

**Deadline:** 9 Oktober 2026, 11:45pm PDT (≈ 10 Okt 13:45 WIB).
**Aturan utama:** jangan menambah fitur baru di luar scope yang sudah dikunci.
Kedalaman eksekusi > banyaknya fitur.

## Scope terkunci

**MVP (wajib selesai):**
- 26 huruf alfabet BISINDO/SIBI dikenali real-time dari webcam
- Teks hasil terjemahan tampil + bisa diucapkan (Web Speech API, `id-ID`)
- Seluruh inference di sisi klien (privasi by design)
- Mode "Latih" untuk menambah data sendiri lewat `capture.html`

**Stretch (hanya bila MVP aman pada 7 Okt malam):**
- 10 frasa layanan publik ("saya demam", "dimana ruang tunggu?", dst.)
- Mode dua arah: petugas mengetik → ditampilkan besar / diucapkan

---

## Hari per hari

### 5 Okt — Data & pipeline
- [ ] Baca [`DATA.md`](DATA.md) — spesifikasi data, katalog sumber, protokol rekam, etika
- [ ] Setup venv, install dependency, jalankan smoke test sintetis
- [ ] Uji `web/capture.html` → rekam 2–3 kelas dulu untuk memastikan alurnya nyaman
- [ ] Unduh 1 dataset publik kandidat + verifikasi lisensinya (catat di `ATTRIBUTIONS.md`)
- [ ] Target hari ini: pipeline `rekam → audit → train (SI) → demo browser` terbukti end-to-end

### 6 Okt — Produksi data & model
- [ ] Rekam 3 periset × 26 huruf × ≥ 2 sesi (target ≥ 300 sampel/kelas; minimal 150 dulu)
- [ ] `python scripts/audit_dataset.py` → bersihkan duplikat/outlier/kelas timpang
- [ ] Latih model, bandingkan KNN vs MLP vs RF
- [ ] **Checkpoint akurasi ≥ 85% pada split signer-independent** (`--group-column signer`)
- [ ] Analisis kelas yang paling sering tertukar → tambah variasi data untuk kelas itu

### 7 Okt — UI/UX
- [ ] Polish tampilan (gunakan referensi Mobbin untuk pola UI yang matang)
- [ ] Uji dengan 2–3 orang lain (beda ukuran tangan, pencahayaan, jarak)
- [ ] **Keputusan go/no-go scope stretch** malam ini

### 8 Okt — Validasi & demo
- [ ] Uji dengan pengguna asli (idealnya komunitas Tuli / relawan)
- [ ] Rekam video demo 2–3 menit (lihat struktur di `devpost-submission.md`)
- [ ] Screenshot 3–5 layar utama untuk submission
- [ ] Rapikan README + atribusi + push ke GitHub (public)

### 9 Okt — Submission
- [ ] Tulis deskripsi Devpost lengkap (problem, solusi, fitur, teknologi, target user)
- [ ] Upload file: video + screenshot
- [ ] Isi link repo & live demo (GitHub Pages / Vercel / Netlify)
- [ ] **Submit sebelum siang hari — jangan mepet deadline**
- [ ] Cek ulang: semua anggota tim terdaftar, semua field terisi

---

## Aturan de-risking

| Jika… | Maka… |
|---|---|
| Akurasi 26 huruf < 85% pada 7 Okt malam | Potong scope: **mode fingerspelling huruf A–J saja**, tapi mulus |
| Kamera/CDN MediaPipe bermasalah saat rekam demo | Siapkan rekaman layar cadangan + mode demo dengan dataset statis |
| Dataset eksternal tidak jelas lisensinya | **Jangan pakai.** Andalkan rekaman sendiri (lebih orisinal pula) |
| Waktu habis untuk UI | Prioritaskan fungsi + dokumentasi; UI cukup bersih dan jelas |

## Catatan penting untuk penilaian

- **Technical Implementation (30%)** — tekankan bahwa classifier dilatih sendiri
  (bukan sekadar memanggil API), lengkap dengan metrik evaluasi di `reports/`.
- **Creativity (20%)** — kebaruannya ada pada konteks layanan publik Indonesia +
  mode dua arah, bukan pada "deteksi tangan" yang sudah umum.
- **Impact (20%)** — sertakan angka: jumlah pengguna BISINDO, kesenjangan akses
  layanan publik. **Verifikasi setiap angka dari sumber resmi sebelum ditulis.**
- **Design & UX (15%)** — satu alur utama yang jelas; jangan menumpuk fitur di layar.
- **Presentation (15%)** — video demo ≤ 3 menit, masalah disebut di 15 detik pertama.

## Yang harus dihindari

- Klaim palsu ("sudah dipakai 1000 orang") — juri akan mempertanyakan, dan ini
  melanggar integritas submission.
- Menyalin repo GitHub utuh tanpa atribusi — lihat `ATTRIBUTIONS.md`.
- Menyentuh scope baru di hari ke-4 atau ke-5.
