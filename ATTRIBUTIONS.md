# Sasmita — Atribusi & Lisensi

Proyek ini dibangun di atas karya open-source berikut. Dicantumkan sesuai ketentuan
lisensi masing-masing, sekaligus sebagai transparansi untuk submission Devpost.

## Komponen pihak ketiga

| Komponen | Digunakan untuk | Sumber | Lisensi |
|---|---|---|---|
| MediaPipe Hand Landmarker (Google) | Deteksi 21 titik landmark tangan, runtime di browser & ekstraksi dataset | https://github.com/google-ai-edge/mediapipe | Apache-2.0 |
| Model `hand_landmarker.task` | Bobot pretrained landmark | https://storage.googleapis.com/mediapipe-models/hand_landmarker/ | Apache-2.0 (Google) |
| scikit-learn | Training & evaluasi classifier (KNN, MLP, RandomForest) | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause |
| NumPy | Komputasi array | https://github.com/numpy/numpy | BSD-3-Clause |
| OpenCV | Membaca gambar/video untuk ekstraksi landmark | https://github.com/opencv/opencv | Apache-2.0 |

## Referensi yang dipelajari

Repo berikut dipakai sebagai **referensi arsitektur**, bukan disalin:

- `khairul3/BISINDO-Sign-Language-Recognition` — pendekatan pengenalan BISINDO
- `daf2a/Sign_Language_Translator_Web_App` — pola MediaPipe di web app
- `gabguerin/Sign-Language-Recognition--MediaPipe-DTW` — landmark → klasifikasi

Seluruh kode di repo ini ditulis sendiri oleh tim Sasmita, kecuali yang
disebutkan pada tabel di atas.

## Dataset

- Dataset isyarat BISINDO: _sumber dan lisensinya harus diisi sendiri setelah
  diunduh dan diverifikasi_ (kandidat: Mendeley Data "BISINDO Sign Language
  Recognition"). **Jangan** memakai dataset tanpa memeriksa lisensinya, dan
  jangan mengklaim data yang tidak benar-benar dipakai.
- Sampel yang direkam sendiri melalui `web/capture.html`: milik tim, boleh
  dirilis di repo ini.

## Lisensi proyek ini

Kode Sasmita dirilis di bawah **MIT License** — lihat `LICENSE`.
