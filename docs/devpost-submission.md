# Devpost Submission Draft — Jemari

> Working draft in English (international judging panel). Copy each section
> into the corresponding Devpost field. Every number below is real and
> traceable to `reports/` in the repo.

---

## Project Title

**Jemari** — turning finger signs into spoken words

*Jemari* means "fingers" in Bahasa Indonesia — because that is where sign
language lives. Jemari reads finger-spelled BISINDO (Indonesian Sign Language)
letters through a webcam and speaks them out loud, entirely on-device.

*(Fallback names if another entry uses it: "TanganBicara", "Sasmita")*

---

## Inspired by

ML Empowerment Build Challenge 3.0 — open theme (real-world problem / social impact).

---

## What it does

Jemari translates BISINDO alphabet hand signs into text and speech in real time,
directly in the browser. A user opens the camera, spells a word letter by letter,
and each recognized letter appears on screen — then the full word can be spoken
aloud with a tap (the spoken output is Bahasa Indonesia, because the person at
the other side of the counter is Indonesian).

Everything runs on the user's device: no server, no account, no API keys, and no
video ever leaves the device. It also ships with a **dataset recorder**
(`web/capture.html`): anyone can record new sign samples with per-signer
metadata, export them, and retrain the model — making the dataset itself a
community-growable artifact.

---

## Problem Statement

At public service counters in Indonesia — community health clinics, village
offices, banks, schools — Deaf signers and hearing staff rarely share a language.
Certified sign language interpreters are scarce and almost never available on
demand, especially outside big cities. The fallback (writing on paper) is slow,
exhausting, and unequal: explaining a symptom or processing a document becomes
an ordeal.

Something as simple as fingerspelling a name or a basic need at a counter
should not require another person to be present.

> `[Add a verified statistic here: number of Deaf sign language users in
> Indonesia and interpreter availability — cite an official source such as
> Kemensos / Gerkatin / WHO before submitting.]`

---

## Solution Overview

Jemari turns any phone or laptop camera into an always-available fingerspelling
bridge:

1. **Hand detection** — MediaPipe Hand Landmarker extracts 21 3D keypoints per
   frame, running in-browser via WebAssembly/WebGL (CPU fallback included).
2. **Normalization** — keypoints are translated to the wrist and scaled by hand
   size, making the features invariant to hand position in frame and to hand
   size (children vs adults).
3. **Classification** — the resulting 63-dimensional vector feeds a small MLP
   that **we trained ourselves** (not a third-party API). Predictions are
   smoothed over a temporal window with a confidence threshold so letters don't
   flicker.
4. **Two-way output** — recognized letters build a transcript that can be spoken
   aloud via the Web Speech API.

Because the whole pipeline is client-side, it works in a waiting room with poor
connectivity and sends no personal data anywhere.

---

## Key Features

- **Real-time sign-to-text** from the webcam, zero installation
- **Indonesian voice output** (Web Speech API, `id-ID`) so counter staff understand
- **Sentence transcript** with copy and clear actions
- **Per-letter confidence indicator** — users can see when the system is unsure
- **100% on-device** — video never leaves the device; no server, no API keys
- **Dataset recorder with signer metadata** — grow the dataset as a community,
  with honest signer-independent evaluation built in
- **Temporal smoothing** — multi-frame voting + confidence thresholding

---

## Technologies Used

| Technology | Role |
|---|---|
| MediaPipe Tasks Vision (Hand Landmarker) | 21 hand keypoints in-browser (WASM/WebGL) |
| JavaScript (ES modules, Canvas API) | Runtime inference, UI, landmark overlay |
| Web Speech API | Indonesian text-to-speech output |
| Python 3.12 + scikit-learn | Training & evaluation (KNN, MLP, RandomForest) |
| NumPy / OpenCV | Data preprocessing, landmark extraction from images |
| localStorage | Client-side dataset storage in the recorder |
| GitHub Pages | Static hosting (HTTPS, required for camera access) |

Credits for open-source components and datasets (with licenses): `ATTRIBUTIONS.md`.

---

## Model & Evaluation

**Data** — 510 landmark samples, 26 BISINDO letter classes. Source: the public
"BISINDO Hand-Sign Detection" dataset (rhiosutoyo, **MIT License**; used in an
IEEE ICRAIE 2023 paper), extracted with MediaPipe Hand Landmarker — hands were
detected in 510 of 520 images (98%). Plus 880 real-anchored synthetic samples
(in-class mixup + per-class PCA sampling) used for **training only**; the test
set is always 100% real data.

**Model** — MLP (128–64) over a StandardScaler; all inference client-side
(MediaPipe WASM for landmarks, plain JavaScript for the classifier).

**Results** — **92.2%** accuracy on a 20% hold-out (macro-F1 90.3%).
Full sweep of all 510 gesture images through the *browser inference path*
(the exact code that runs in the live demo): **93.9%**.

**Per letter (510-image verification):** 21 of 26 letters ≥ 95%; weakest:
B 65%, H 74%, K 80%, M 80%. The remaining errors — B→E, K→P, M→N, H→D — are
genuinely near-identical handshapes in BISINDO, not random failures.

**Configuration comparison** (controlled benchmark, identical split for every
configuration):

| Configuration | Model | Accuracy | Δ vs baseline | Model size |
|---|---|---:|---:|---:|
| baseline | KNN | 84.3% | — | 568 KB |
| baseline | MLP | 84.3% | — | 340 KB |
| augmentation 3× | MLP | 89.2% | +4.9 pts | 366 KB |
| **synthetic** | **MLP** | **92.2%** | **+7.9 pts** | **366 KB** |

**Automated data audit**: balanced classes (1.33× ratio), 1 duplicate and 1
outlier detected and handled; the closest class centroids match the model's
actual confusion pairs.

**Acknowledged limitation** — the public dataset covers a single signer, so the
numbers above use a random split rather than a signer-independent one.
Signer-independent evaluation is the immediate next validation step; the
infrastructure is already built (per-signer metadata + `StratifiedGroupKFold` in
`scripts/train.py`, plus a controlled benchmark harness).

---

## Target Users

1. **Deaf BISINDO signers** — especially those who deal with public services
   without an interpreter.
2. **Public service staff** — clinic officers, village administrators, bank
   tellers, teachers — who need to communicate without knowing sign language.
3. **Families & volunteers** — learning basic signs via the built-in recorder
   and reference images.

---

## Social Impact Statement (optional but encouraged)

`[Write 3–4 sentences: the access gap at public counters in Indonesia, who is
helped, and why a free, offline-capable, privacy-preserving tool matters.
Include only verified statistics with sources.]`

---

## Project Files (at least 1 required)

- **Demo video, 2–3 minutes** — structure:
  1. 0:00–0:15 — the problem (one sentence + a counter scenario)
  2. 0:15–0:45 — live demo: sign → letters → word → spoken output
  3. 0:45–1:30 — how it works (landmark overlay, on-device pipeline)
  4. 1:30–2:15 — dataset recorder + model evaluation numbers (show real metrics)
  5. 2:15–2:45 — impact & next steps
- **3–5 screenshots:** main demo, landmark overlay, recorder with reference
  images, dataset table, metrics report
- Record at ≥720p, keep hands clearly visible, and **show the transcript on
  screen** (judges may watch without sound)

## Project Link / Repository

- **Live demo:** https://alvnvnc.github.io/bisindo-translator/ (HTTPS — camera &
  TTS work; try it on laptop or phone)
- **GitHub repo (public):** https://github.com/Alvnvnc/bisindo-translator
- The trained `model.json` and landmark dataset ship with the site — the demo
  runs fully with no backend

## Team Details

| Name | Role |
|---|---|
| `[…]` | `[e.g. ML — dataset, training, evaluation]` |
| `[…]` | `[e.g. Frontend — UI, MediaPipe integration]` |
| `[…]` | `[e.g. User research, documentation, video]` |

*(Solo submissions are allowed — one row is enough.)*

---

## Pre-submit checklist

- [ ] Every field filled; no `[placeholders]` left
- [ ] Every statistic has a verifiable source
- [ ] Video & screenshots uploaded and playable
- [ ] Repo link works and is public
- [ ] Live demo opens on someone else's phone (HTTPS, camera works)
- [ ] `ATTRIBUTIONS.md` lists all third-party components
- [ ] Team members added to the submission
- [ ] Submit well before the deadline (Oct 9, 11:45pm PDT)
