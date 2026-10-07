# Jemari — Attributions & Licenses

This project builds on the following open-source work. Credits are listed as
required by each license, and for transparency toward the hackathon judges.

## Third-party components

| Component | Used for | Source | License |
|---|---|---|---|
| MediaPipe Hand Landmarker (Google) | 21 hand keypoints, browser runtime & dataset extraction | https://github.com/google-ai-edge/mediapipe | Apache-2.0 |
| `hand_landmarker.task` model | Pretrained landmark weights | https://storage.googleapis.com/mediapipe-models/hand_landmarker/ | Apache-2.0 (Google) |
| scikit-learn | Classifier training & evaluation (KNN, MLP, RandomForest) | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause |
| NumPy | Array computation | https://github.com/numpy/numpy | BSD-3-Clause |
| OpenCV | Reading images/videos for landmark extraction | https://github.com/opencv/opencv | Apache-2.0 |

## Datasets

| Dataset | Used for | Source | License |
|---|---|---|---|
| BISINDO Hand-Sign Detection Dataset (rhiosutoyo) | 510 landmark samples (26 letters), used in an IEEE ICRAIE 2023 paper | https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset | **MIT** — citation below |
| Self-recorded samples (`web/capture.html`) | Community-growable dataset | This repository | CC BY 4.0 (ours) |

Citation for the dataset:

```bibtex
@INPROCEEDINGS{10468194,
  author  = {Joan, David and Vincent, Vincent and Daniel, Kevin Jason and
             Achmad, Said and Sutoyo, Rhio},
  booktitle = {2023 IEEE 8th International Conference on Recent Advances
               and Innovations in Engineering (ICRAIE)},
  title   = {BISINDO Hand-Sign Detection Using Transfer Learning},
  year    = {2023},
  pages   = {1-7},
  doi     = {10.1109/ICRAIE59459.2023.10468194}
}
```

The reference thumbnails in `web/ref/` are derived from the same MIT-licensed
dataset (resized copies, 3 per letter).

## Repositories studied as references

Used as **architectural references**, not copied:

- `khairul3/BISINDO-Sign-Language-Recognition` — BISINDO recognition approach
- `daf2a/Sign_Language_Translator_Web_App` — MediaPipe-in-webapp patterns
- `gabguerin/Sign-Language-Recognition--MediaPipe-DTW` — landmarks → classification

All code in this repository was written by the Jemari team, except the
components listed in the tables above.

## License of this project

The Jemari code is released under the **MIT License** — see `LICENSE`.
