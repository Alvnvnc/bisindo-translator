# Security & Privacy — Jemari

## No secrets, by design

This application is **100% client-side**:

- **No API keys, tokens, or secrets** anywhere in the code, configuration, or CI.
- **No backend server** of our own — it is a static site.
- The only network requests at runtime:
  1. MediaPipe Tasks Vision (WASM + landmark model) from the public jsDelivr CDN
     and `storage.googleapis.com` (public Google-hosted assets),
  2. loading `model.json` and reference thumbnails from this same repository.

Because there is nothing secret to protect, the attack surface is minimal:
what we protect is not a key, but **the user in front of the camera**.

## User privacy

- **Camera video never leaves the device.** All landmark detection and
  classification runs locally (WebAssembly/WebGL + JavaScript).
- **No telemetry, tracking cookies, or analytics.**
- The dataset recorder (`web/capture.html`) stores **63-number vectors** in the
  browser's `localStorage`; no images or faces are ever stored, and CSV export
  only happens when the user presses the button.

## Reporting a vulnerability

Open an issue in this repository, or contact the repo owner via GitHub.
Please include reproduction steps. Response within ~48 hours.

## Hosting environment notes

GitHub Pages does not allow custom HTTP headers (e.g. a `Content-Security-Policy`
header). The chosen mitigations:

- The pages process no sensitive user data (see Privacy).
- All third-party assets load from official CDNs (jsDelivr, Google storage) with
  URLs pinned in `web/hands.js`.
- There are no forms, and no input is ever submitted anywhere.
