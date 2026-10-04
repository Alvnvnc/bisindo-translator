/**
 * Alat perekam dataset: webcam → landmark → localStorage → CSV.
 *
 * Setiap sampel menyimpan metadata sesuai teori pembuatan dataset:
 *   - f   : 63 fitur landmark ternormalisasi
 *   - ses : ID sesi rekaman (satu sesi = satu kali tekan Rekam)
 *   - s   : ID periset/signer
 *
 * Metadata ini bukan hiasan — tanpa ID periset, model tidak bisa dievaluasi
 * secara signer-independent (diuji pada orang yang belum pernah dilihat),
 * sehingga akurasi yang dilaporkan akan menyesatkan.
 *
 * CSV yang dihasilkan identik formatnya dengan scripts/extract_landmarks.py,
 * jadi data rekaman sendiri dan dataset gambar bisa digabung untuk training.
 */

import { normalizeLandmarks } from "./landmarks.js";
import { createHandLandmarker, startCamera, stopCamera, drawHand } from "./hands.js";

const STORAGE_KEY = "handtalk.samples.v1";
const SIGNER_KEY = "handtalk.signer";
const FEATURE_DIM = 63;
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const $ = (id) => document.getElementById(id);

const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");

const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const recordBtn = $("recordBtn");
const classInput = $("classInput");
const signerInput = $("signerInput");
const frameCountInput = $("frameCount");
const progressFill = $("progressFill");
const recordStatus = $("recordStatus");
const camStatus = $("camStatus");
const recordHint = $("recordHint");
const classRows = $("classRows");
const totalStatus = $("totalStatus");
const exportBtn = $("exportBtn");
const importInput = $("importInput");
const clearBtn = $("clearBtn");

const state = {
  landmarker: null,
  running: false,
  recording: false,
  rafId: null,
  lastVideoTime: -1,
  latestLandmarks: null,
};

/* ------------------------------ penyimpanan ------------------------------ */

function newSessionId() {
  const now = new Date();
  const stamp = now.toISOString().slice(0, 16).replace(/[-:T]/g, "");
  const rand = Math.random().toString(36).slice(2, 6);
  return `ses-${stamp}-${rand}`;
}

function loadStore() {
  let raw;
  try {
    raw = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }

  // Migrasi format lama (array fitur polos) → format ber-metadata.
  const store = {};
  for (const [label, entries] of Object.entries(raw)) {
    store[label] = entries.map((e) =>
      Array.isArray(e) ? { f: e, ses: "legacy", s: "unknown" } : e,
    );
  }
  return store;
}

function saveStore(store) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

function addSamples(label, samples, session, signer) {
  const store = loadStore();
  store[label] = (store[label] || []).concat(
    samples.map((f) => ({ f, ses: session, s: signer })),
  );
  saveStore(store);
  renderTable();
}

function deleteClass(label) {
  const store = loadStore();
  delete store[label];
  saveStore(store);
  renderTable();
}

function renderTable() {
  const store = loadStore();
  const labels = Object.keys(store).sort();

  classRows.innerHTML = "";
  let total = 0;

  for (const label of labels) {
    const entries = store[label];
    const n = entries.length;
    total += n;
    const signers = new Set(entries.map((e) => e.s || "unknown"));
    const signerInfo = signers.size ? `${signers.size} periset` : "tanpa metadata";

    const tr = document.createElement("tr");
    const pillClass = n >= 300 ? "pill" : "pill pill-warn";
    tr.innerHTML = `
      <td><strong>${escapeHtml(label)}</strong><br /><span class="hint">${escapeHtml(signerInfo)}</span></td>
      <td class="num"><span class="${pillClass}">${n}</span></td>
      <td class="num"><button class="btn btn-small btn-danger" data-del="${escapeHtml(label)}">Hapus</button></td>
    `;
    classRows.appendChild(tr);
  }

  totalStatus.textContent = labels.length
    ? `${labels.length} kelas · ${total} sampel total`
    : "Belum ada data. Rekam kelas pertama Anda.";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

classRows.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-del]");
  if (!btn) return;
  const label = btn.dataset.del;
  if (confirm(`Hapus semua sampel kelas "${label}"?`)) deleteClass(label);
});

/* -------------------------------- kamera -------------------------------- */

async function start() {
  startBtn.disabled = true;
  camStatus.textContent = "Menyiapkan kamera & model MediaPipe…";
  try {
    if (!state.landmarker) state.landmarker = await createHandLandmarker();
    await startCamera(video);
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    state.running = true;
    stopBtn.disabled = false;
    recordBtn.disabled = false;
    camStatus.textContent = "Kamera aktif.";
    camStatus.className = "status ok";
    recordHint.textContent = "Pilih kelas lalu tekan Rekam.";
    loop();
  } catch (err) {
    console.error(err);
    startBtn.disabled = false;
    camStatus.textContent = `Gagal: ${err.message}`;
    camStatus.className = "status err";
  }
}

function stop() {
  state.running = false;
  if (state.rafId) cancelAnimationFrame(state.rafId);
  stopCamera(video);
  drawHand(ctx, null, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  recordBtn.disabled = true;
  camStatus.textContent = "Kamera dimatikan.";
  camStatus.className = "status";
  recordHint.textContent = "Aktifkan kamera dulu.";
}

function loop() {
  if (!state.running) return;

  if (video.currentTime !== state.lastVideoTime && video.readyState >= 2) {
    state.lastVideoTime = video.currentTime;
    const result = state.landmarker.detectForVideo(video, performance.now());
    state.latestLandmarks = result.landmarks?.[0] || null;
    drawHand(ctx, state.latestLandmarks, overlay.width, overlay.height);
  }

  state.rafId = requestAnimationFrame(loop);
}

/* ------------------------------- rekaman -------------------------------- */

function setStatus(text, kind = "") {
  recordStatus.textContent = text;
  recordStatus.className = `status ${kind}`;
}

async function record() {
  const label = classInput.value.trim().toUpperCase();
  if (!label) {
    setStatus("Isi nama kelas dulu (mis. A).", "err");
    classInput.focus();
    return;
  }

  const signer = signerInput.value.trim() || "unknown";
  localStorage.setItem(SIGNER_KEY, signer);

  const target = Math.max(3, Math.min(60, parseInt(frameCountInput.value, 10) || 12));
  const session = newSessionId();
  const collected = [];
  state.recording = true;
  recordBtn.disabled = true;
  progressFill.style.width = "0%";
  setStatus(`Merekam ${target} frame… gerakkan tangan perlahan.`);

  const MIN_INTERVAL_MS = 90; // jeda minimum antar sampel
  const MIN_DISTANCE = 0.004; // buang duplikat yang nyaris identik
  const NO_HAND_LIMIT_MS = 1500;

  const startedAt = performance.now();
  const timeoutMs = target * 220 + 4000;
  let lastCaptureAt = 0;
  let lastHandAt = performance.now();

  await new Promise((resolve) => {
    const tick = () => {
      const now = performance.now();
      if (collected.length >= target) return resolve();

      if (now - startedAt > timeoutMs) {
        setStatus(
          collected.length
            ? `Waktu habis — ${collected.length}/${target} frame terekam.`
            : "Waktu habis — tidak ada frame terekam. Pastikan tangan terlihat dan bergerak.",
          collected.length ? "" : "err",
        );
        return resolve();
      }

      if (state.latestLandmarks) {
        lastHandAt = now;
        const current = normalizeLandmarks(state.latestLandmarks);
        const last = collected[collected.length - 1];

        // Terima bila cukup waktu berlalu dan pose-nya tidak identik dengan sebelumnya.
        if (now - lastCaptureAt >= MIN_INTERVAL_MS && (!last || distance(last, current) > MIN_DISTANCE)) {
          collected.push(current);
          lastCaptureAt = now;
          progressFill.style.width = `${(collected.length / target) * 100}%`;
          setStatus(`Merekam… ${collected.length}/${target} — ubah posisi tangan sedikit.`);
        } else if (now - lastCaptureAt > 700) {
          setStatus(`Merekam… ${collected.length}/${target} — tangan terlalu diam, geser perlahan.`);
        }
      } else if (now - lastHandAt > NO_HAND_LIMIT_MS) {
        setStatus("Tangan tidak terdeteksi — tampilkan tangan ke kamera.", "err");
        return resolve();
      }

      requestAnimationFrame(tick);
    };
    tick();
  });

  state.recording = false;
  recordBtn.disabled = false;
  progressFill.style.width = "0%";

  if (collected.length) {
    addSamples(label, collected, session, signer);
    setStatus(`${collected.length} sampel → kelas "${label}" · sesi ${session} · periset ${signer}.`, "ok");
    classInput.value = nextClass(label);
  }
}

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/** Usulkan kelas berikutnya agar perekaman alfabet berjalan berurutan. */
function nextClass(label) {
  const idx = ALPHABET.indexOf(label);
  return idx >= 0 && idx < ALPHABET.length - 1 ? ALPHABET[idx + 1] : label;
}

/* ------------------------------ ekspor/impor ----------------------------- */

const EXTRA_COLUMNS = ["session", "signer"];

function exportCsv() {
  const store = loadStore();
  const labels = Object.keys(store).sort();
  if (!labels.length) {
    setStatus("Belum ada data untuk diekspor.", "err");
    return;
  }

  const header = [
    "class",
    ...Array.from({ length: FEATURE_DIM }, (_, i) => `f${i}`),
    ...EXTRA_COLUMNS,
  ];
  const lines = [header.join(",")];
  let total = 0;

  for (const label of labels) {
    for (const entry of store[label]) {
      lines.push(
        [csvEscape(label), ...entry.f.map((v) => v.toFixed(6)), csvEscape(entry.ses), csvEscape(entry.s)].join(","),
      );
      total += 1;
    }
  }

  const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  downloadBlob(new Blob([lines.join("\n")], { type: "text/csv" }), `handtalk_samples_${stamp}.csv`);
  setStatus(`${total} sampel diekspor (dengan session & signer). Simpan ke data/samples/.`, "ok");
}

function csvEscape(value) {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function importCsv(file) {
  const text = await file.text();
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) {
    setStatus("File CSV kosong.", "err");
    return;
  }

  const header = lines[0].split(",");
  if (header[0] !== "class" || header.length < FEATURE_DIM + 1) {
    setStatus("Format CSV tidak dikenali (butuh kolom class,f0..f62[,session,signer]).", "err");
    return;
  }

  const sessionIdx = header.indexOf("session");
  const signerIdx = header.indexOf("signer");

  const store = loadStore();
  let added = 0;

  for (const line of lines.slice(1)) {
    const cells = line.split(",");
    if (cells.length !== header.length) continue;
    const label = cells[0].replace(/^"|"$/g, "").trim().toUpperCase();
    const feats = cells.slice(1, FEATURE_DIM + 1).map(Number);
    if (!label || feats.some(Number.isNaN)) continue;

    const ses = sessionIdx >= 0 ? cells[sessionIdx].replace(/^"|"$/g, "") : "imported";
    const signer = signerIdx >= 0 ? cells[signerIdx].replace(/^"|"$/g, "") : "unknown";

    store[label] = (store[label] || []).concat([{ f: feats, ses, s: signer }]);
    added += 1;
  }

  saveStore(store);
  renderTable();
  setStatus(`${added} sampel diimpor dari ${file.name}.`, added ? "ok" : "err");
}

/* -------------------------------- wiring -------------------------------- */

$("classList").innerHTML = ALPHABET.map((c) => `<option value="${c}">`).join("");

signerInput.value = localStorage.getItem(SIGNER_KEY) || "";

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
recordBtn.addEventListener("click", record);
exportBtn.addEventListener("click", exportCsv);
importInput.addEventListener("change", (e) => {
  if (e.target.files?.[0]) importCsv(e.target.files[0]);
  e.target.value = "";
});
clearBtn.addEventListener("click", () => {
  if (confirm("Hapus seluruh dataset tersimpan di browser? Ekspor dulu bila perlu.")) {
    localStorage.removeItem(STORAGE_KEY);
    renderTable();
    setStatus("Semua data dihapus.", "ok");
  }
});

classInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !recordBtn.disabled) record();
});

video.addEventListener("resize", () => {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
});

renderTable();
