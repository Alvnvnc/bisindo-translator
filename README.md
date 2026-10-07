# Jemari

**From fingertip signs to spoken words.**

🔗 **Live demo: https://alvnvnc.github.io/bisindo-translator/**

*Jemari* ("fingers" in Bahasa Indonesia) is a real-time BISINDO (Indonesian
Sign Language) alphabet translator that turns webcam hand signs into text and
speech — running entirely on the user's device. Built as a communication bridge
at public service counters (clinics, village offices, banks, schools), for
ML Empowerment Build Challenge 3.0.

- **Submission deadline:** October 9, 2026, 11:45pm PDT
- **Challenge:** [ML Empowerment Build Challenge 3.0](https://ml-build-challenge-3.devpost.com/)

> The spoken output is deliberately **Bahasa Indonesia** (`id-ID`): the person
> listening at the counter is Indonesian. The interface is in English for the
> international demo.

---

## Architecture

```
Browser camera
      │
      ▼
MediaPipe Hand Landmarker (WebAssembly, on-device)
      │  21 keypoints per hand (x, y, z)
      ▼
Normalization (translate to wrist + scale by hand size)
      │  63-dimensional feature vector
      ▼
Self-trained classifier (KNN / small MLP) ── loaded from web/model.json
      │
      ▼
Temporal smoothing (N-frame voting + confidence threshold)
      │
      ▼
On-screen text  +  speech (Web Speech API, id-ID)
```

All inference runs **client-side**. Camera frames never leave the user's
device — privacy by design is part of the product's value.

Development workflow:

1. **Collect data** — `web/capture.html` (record landmark samples with per-signer
   metadata), or `scripts/extract_landmarks.py` (convert an image dataset to the
   same format).
2. **Train** — `scripts/train.py` → evaluation + export to `web/model.json`.
3. **Run the demo** — open `web/index.html`.

---

## Setup

Requires Python 3.12 (via `uv`); Node.js is only needed for the optional
local parity test.

```bash
# 1. Create a Python 3.12 virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. One-time pipeline smoke test with synthetic data
python scripts/make_sample_dataset.py
python scripts/audit_dataset.py     # data quality checklist
python scripts/train.py             # default split: signer-independent

# 4. Verify Python ↔ JavaScript produce identical numbers
python scripts/check_parity.py

# 5. (Optional) Controlled benchmark: baseline vs augmentation vs synthetic
python scripts/benchmark.py

# 6. Run the web demo
python -m http.server 8000
# open http://localhost:8000/web/index.html       → translation demo
# open http://localhost:8000/web/capture.html     → dataset recorder
```

> `http.server` is required because MediaPipe Tasks Vision fetches its model
> over HTTP, which `file://` blocks. For the public demo, we deploy to GitHub
> Pages (HTTPS is required for camera access).

---

## Repository structure

```
├── README.md
├── ATTRIBUTIONS.md            # third-party credits + licenses
├── SECURITY.md                # security & privacy posture
├── LICENSE                    # MIT
├── requirements.txt
├── index.html                 # landing page (production)
├── data/
│   ├── raw/                   # raw images per class (gitignored)
│   └── samples/               # landmark CSVs: class,f0..f62,session,signer
├── scripts/
│   ├── extract_landmarks.py   # images/video → landmark CSV (MediaPipe)
│   ├── audit_dataset.py       # data quality audit before training
│   ├── generate_synthetic.py  # real-anchored synthetic data (mixup/pca)
│   ├── benchmark.py           # controlled config comparison (same split)
│   ├── train.py               # train + evaluate (SI split) + export model
│   ├── make_sample_dataset.py # synthetic toy data for pipeline smoke tests
│   ├── check_parity.py        # Python ↔ JS parity guard
│   └── check_parity.mjs
├── web/
│   ├── index.html             # main demo: real-time translation
│   ├── app.js                 # camera → classification → UI + TTS wiring
│   ├── capture.html           # webcam dataset recorder
│   ├── capture.js
│   ├── landmarks.js           # normalization + inference (the math core)
│   ├── hands.js               # MediaPipe Tasks Vision wrapper
│   ├── ref/                   # reference thumbnails per letter + manifest
│   ├── model.json             # trained model (committed so the demo deploys)
│   └── style.css
├── reports/                   # training metrics & audits (gitignored)
└── docs/
    ├── DATA.md                # data preparation guide (in Bahasa Indonesia)
    ├── PLAN.md                # 5-day sprint plan + de-risking rules (id)
    └── devpost-submission.md  # English submission draft
```

---

## Data format

All data sources converge on one CSV format in `data/samples/`:

```
class,f0,f1,...,f62,session,signer
A,0.123,-0.456,...,ses-20261005-1200-abc,S01
```

- `class` — sign label (e.g. letters `A`–`Z`)
- `f0..f62` — 21 keypoints × 3 coordinates (x, y, z), **already normalized**
- `session` — recording session id (one press of "Record"); optional column
- `signer` — signer id; **required for honest signer-independent evaluation**

`session`/`signer` are read by header name, so legacy CSVs without them still
work. The full data-preparation methodology (where to find datasets, recording
protocol, labeling conventions, ethics) is in
[`docs/DATA.md`](docs/DATA.md) *(in Bahasa Indonesia)*.

Normalization (must stay identical between Python and JS — see
`scripts/train.py` and `web/app.js`):

1. Translate all points so the wrist (keypoint 0) is at the origin.
2. Divide by the farthest wrist-to-keypoint distance (scale invariance).
3. Flatten into a 63-dimensional vector.

This makes the model independent of hand position in frame and of hand size.

---

## Status

- [x] Pipeline scaffold: landmarks → training → export → browser inference
- [x] Dataset recorder (`web/capture.html`) with signer/session metadata
- [x] Automated data audit (`scripts/audit_dataset.py`)
- [x] Real-anchored synthetic data (`scripts/generate_synthetic.py`) — train only
- [x] Signer-independent split & augmentation per theory (`scripts/train.py`)
- [x] Controlled benchmark (`scripts/benchmark.py`)
- [x] Python ↔ JS parity test (`scripts/check_parity.py`) — passing
- [x] Core demo UI: big letter, confidence meter, transcript, `id-ID` TTS
- [x] **Real data: 510 landmarks from a public MIT-licensed BISINDO dataset**
- [x] **Model v1: MLP, 26 letters, 92.2% (random split, single signer)**
- [x] Verified against all 510 gesture images through the browser path: 93.9%
- [x] Landing page + production deploy (GitHub Pages, HTTPS)
- [ ] New signer recordings → honest signer-independent evaluation
- [ ] Real user testing + demo video
- [ ] Practice mode & service phrases (stretch)

> ℹ️ `data/samples/rhiosutoyo_dataset.csv` is a landmark snapshot extracted from
> the MIT-licensed public dataset. `web/model.json` was trained on it plus 880
> synthetic samples (training only). The 92.2% figure uses a random split
> because that dataset has a single signer — honest signer-independent numbers
> await new signer recordings.

Daily plan in [`docs/PLAN.md`](docs/PLAN.md), data methodology in
[`docs/DATA.md`](docs/DATA.md), submission draft in
[`docs/devpost-submission.md`](docs/devpost-submission.md).
