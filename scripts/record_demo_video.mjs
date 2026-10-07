/**
 * Rekam Video 1 — tur layar Jemari (tanpa suara; voiceover dari Video 2).
 *
 * Trik: kamera palsu Chromium diisi feed y4m yang mengeja H-A-L-O dari foto
 * dataset (lihat docs/video-script.md), sehingga rekaman menunjukkan huruf
 * yang BENAR-BENAR dikenali model melalui pipeline kamera.
 *
 * Pemakaian (butuh playwright global + feed y4m):
 *   ffmpeg tidak wajib; konversi webm→mp4 terpisah.
 *   node record_demo_video.mjs
 */
import { chromium } from "@playwright/test";

const BASE = "https://alvnvnc.github.io/bisindo-translator";
const FEED = "/tmp/opencode/jemari_feed.y4m";
const RAW_DIR = "/tmp/opencode/v1_raw";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({
  headless: true,
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-capture",
    `--use-file-for-fake-video-capture=${FEED}`,
  ],
});

const context = await browser.newContext({
  permissions: ["camera"],
  viewport: { width: 1600, height: 900 },
  recordVideo: { dir: RAW_DIR, size: { width: 1600, height: 900 } },
});

const page = await context.newPage();
const hold = async (s, { wander = true } = {}) => {
  const steps = Math.max(1, Math.round(s * 2));
  for (let i = 0; i < steps; i++) {
    if (wander) await page.mouse.move(300 + Math.random() * 900, 200 + Math.random() * 500);
    await sleep(500);
  }
};

/* --- 1. Landing: hero → scroll pelan → kembali ke atas ------------------- */
await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
await hold(4);
await page.evaluate(() => window.scrollTo({ top: 700, behavior: "smooth" }));
await hold(3);
await page.evaluate(() => window.scrollTo({ top: 1500, behavior: "smooth" }));
await hold(3);
await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
await hold(2);

/* --- 2. Demo: buka, tampilkan UI, nyalakan kamera ------------------------ */
await page.click("a.btn.primary");
await page.waitForLoadState("networkidle");
await hold(3, { wander: false });

await page.click("#startBtn");
await hold(6, { wander: false }); // model & kamera warm-up

/* --- 3. Huruf H-A-L-O muncul (feed 3 detik per huruf) -------------------- */
await hold(30, { wander: false }); // transkrip menyusun "HALOHALO…"

/* --- 4. Speak: ucapkan transkrip ----------------------------------------- */
await page.click("#speakBtn");
await hold(4, { wander: false });

/* --- 5. Recorder: referensi, rekam sampel, tabel dataset ----------------- */
await page.goto(`${BASE}/web/capture.html`, { waitUntil: "networkidle" });
await hold(3);
await page.fill("#classInput", "A");
await page.fill("#signerInput", "S01");
await hold(3, { wander: false }); // foto referensi terlihat
await page.click("#startBtn");
await page.waitForTimeout(4000);
await page.click("#recordBtn");
await hold(5, { wander: false }); // progress bar merekam
await hold(4); // tabel dataset terlihat

/* --- 6. GitHub: README & metrik ------------------------------------------ */
await page.goto("https://github.com/Alvnvnc/bisindo-translator", {
  waitUntil: "domcontentloaded",
});
await hold(3);
await page.evaluate(() => window.scrollTo({ top: 900, behavior: "smooth" }));
await hold(4);
await page.evaluate(() => window.scrollTo({ top: 2200, behavior: "smooth" }));
await hold(4);

/* --- 7. Penutup: tagline di landing -------------------------------------- */
await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
await page.evaluate(() => window.scrollTo({ top: 1550, behavior: "smooth" }));
await hold(6, { wander: false });

await context.close(); // video disimpan saat context ditutup
await browser.close();
console.log("rekaman selesai");
