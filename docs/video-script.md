# Video Script — Jemari (single voiceover, 3 source videos)

**Total target:** ±2:50 after assembly.
**Voiceover:** read once (Video 2), it covers all three sources.
**Assembly:** Video 2 (talking head) + Video 1 (screen capture, produced) +
Video 3 (live hands) cut together on the timecodes below.

| Source | Content | Who |
|---|---|---|
| V1 | Screen capture of the site & tooling (already produced: `jemari_video1.mp4`) | produced |
| V2 | Talking head, reading this script in front of a camera | you |
| V3 | You using the tool live with your own hands | you |

Speaking pace: ~2.5 words/second. **Pause at every `//`.** Slow down on numbers.

---

## SEG 1 — V2 (talking head) · 0:00–0:18 · The problem

> At public service counters in Indonesia — a clinic, a bank, a village office —
> a Deaf customer and a hearing officer often can't understand each other. //
> Sign language interpreters are rare. // Writing everything down is slow,
> and unfair. //
> **So I built something for that moment.**

**On screen:** you, facing camera. (Optional b-roll later: counter photos.)

---

## SEG 2 — V1 (screen) · 0:18–0:30 · The idea

> This is **Jemari**. // It reads BISINDO finger-spelling from a regular webcam,
> and turns it into text and speech — completely on your device.

**V1 cue:** landing page, slow scroll down to the buttons, click **Try the Live Demo**.

---

## SEG 3 — V1 (screen) · 0:30–1:00 · Letters appear

> Watch. // I turn on the camera… the system detects twenty-one hand keypoints…
> and letters appear, one by one, each with a confidence score. //
> The transcript builds a word — and with one tap, the browser speaks it out
> loud, in Indonesian.

**V1 cue:** click **Start camera** → H, A, L, O appear (hold ~20s) → move mouse
to the transcript → click **🔊 Speak**.

---

## SEG 4 — V1 (screen) · 1:00–1:20 · Honest data tooling

> Jemari also ships with a dataset recorder. // Every sample is stored with a
> **signer ID** and a **session ID** — so the model can be evaluated honestly,
> on signers it has never seen. // There's even a reference photo for every
> letter — so anyone can contribute, without knowing sign language.

**V1 cue:** open **Record & Train** → show reference photos panel → type class
`A`, signer `S01` → press **Record** → let the progress bar finish → show the
dataset table.

---

## SEG 5 — V2 (talking head) · 1:20–1:26 · Transition

> Let me show you, live.

**On screen:** you, smiling, turn slightly toward the laptop.

---

## SEG 6 — V3 (live hands) · 1:26–2:05 · The real thing

> Now watch it work on me. // **H…** // **A…** // **L…** // **O…** //
> [letters appear] // …and spoken out loud. //
> Everything you just saw ran on this laptop. // No internet. No API keys.
> And no video ever left this device.

**V3 cues:** frame the shot so **your hand AND the screen** are both visible
(or record the screen, with you visible in a corner window).
Sign each letter **slowly**, hold **2 seconds**. Spelling `H-A-L-O` keeps it in
sync with the script — or spell your own name. After the last letter, click
**🔊 Speak** on screen.

---

## SEG 7 — V1 (screen) · 2:05–2:30 · Under the hood

> Under the hood: MediaPipe extracts twenty-one hand keypoints per frame, //
> and a small neural network — trained by us, on a public, MIT-licensed BISINDO
> dataset — classifies them. // It reached **ninety-two percent** accuracy on
> held-out data, // and we verified every single gesture image through the same
> code that runs in this demo.

**V1 cue:** scroll the GitHub README (architecture diagram → evaluation table),
then show the benchmark table in `reports/`.

---

## SEG 8 — V2 (talking head) · 2:30–2:50 · Closing

> **Jemari** means "fingers". // This is a prototype today — but every piece of
> it is open, documented, and built to grow together with the Deaf community. //
> **From fingertip signs, to spoken words. Thank you.**

**On screen:** you; end card = landing page tagline.

---

## Recording guide — Video 2 (voiceover, talking head)

- Camera at **eye level**, window behind you or plain wall (no backlight).
- Face a light source; avoid ceiling-only light.
- Read **slowly**; each `//` = full stop + breath. It always feels too slow —
  it isn't.
- Record the whole script 2–3 times; we cut the best takes per segment.
- Optional: record in a quiet room; we don't need background music.

## Recording guide — Video 3 (live hands)

- Open the live demo: https://alvnvnc.github.io/bisindo-translator/
- Frame: **your signing hand + the screen in one shot** (phone on a stand,
  landscape), or screen-capture with your face in a small corner window.
- Light your hand from the front; avoid a bright window behind you.
- Sign **H-A-L-O slowly, 2 seconds per letter**, then click 🔊 Speak.
- Do 2–3 full takes.

## Assembly guide

1. Lay Video 2's audio as the single voiceover track.
2. Cut V1 and V3 on the timecodes above (each SEG header is the target start).
3. V1 was recorded without sound by design — V2's voice covers it completely.
4. Add captions (English) for the whole video — judges often watch muted.
5. Export 1080p, MP4 (H.264), ≤ 3:00, then upload to Devpost.
