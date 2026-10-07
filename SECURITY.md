# Keamanan & Privasi — Jemari

## Tanpa rahasia, by design

Aplikasi ini **100% sisi klien (client-side)**:

- **Tidak ada API key, token, atau secret** di kode, konfigurasi, maupun CI.
- **Tidak ada backend/server** milik proyek — hanya berkas statis.
- Satu-satunya permintaan jaringan saat runtime:
  1. MediaPipe Tasks Vision (WASM + model landmark) dari CDN publik jsDelivr
     dan `storage.googleapis.com` (aset publik Google),
  2. memuat `model.json` & thumbnail referensi dari repo ini sendiri.

Karena tidak ada rahasia yang perlu dijaga, permukaan serangannya minimal:
yang dilindungi bukan key, melainkan **pengguna di depan kamera**.

## Privasi pengguna

- **Video kamera tidak pernah meninggalkan perangkat.** Seluruh deteksi landmark
  dan klasifikasi berjalan lokal (WebAssembly/WebGL + JavaScript).
- **Tidak ada telemetri, cookie pelacak, atau analitik.**
- Fitur perekaman dataset (`web/capture.html`) menyimpan **vektor 63 angka**
  di `localStorage` perangkat pengguna; tidak ada gambar/wajah yang disimpan,
  dan ekspor CSV hanya terjadi bila pengguna menekan tombolnya.

## Melaporkan kerentanan

Buka issue di repositori ini, atau hubungi pemilik repo melalui GitHub.
Mohon sertakan langkah reproduksi. Respons dalam ±48 jam.

## Catatan lingkungan hosting

GitHub Pages tidak mengizinkan custom HTTP header (mis. `Content-Security-Policy`
via header). Mitigasi yang dipilih:

- Halaman tidak memproses data sensitif apa pun dari pengguna (lihat Privasi).
- Semua aset pihak ketiga dimuat dari CDN resmi (jsDelivr, Google storage) dengan
  URL yang dipatok di `web/hands.js`.
- Tidak ada form, tidak ada input yang dikirim ke mana pun.
