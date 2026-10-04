/**
 * Bagian JavaScript dari uji paritas — dipanggil oleh scripts/check_parity.py.
 *
 *   node scripts/check_parity.mjs <cases.json> <landmarks.js>
 *
 * Membandingkan hasil normalizeLandmarks() dan predict() dengan nilai harapan
 * yang dihitung di Python.
 */

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [casesPath, landmarksPath] = process.argv.slice(2);
if (!casesPath || !landmarksPath) {
  console.error("Pemakaian: node check_parity.mjs <cases.json> <landmarks.js>");
  process.exit(2);
}

const { normalizeLandmarks, predict } = await import(pathToFileURL(landmarksPath).href);
const cases = JSON.parse(readFileSync(casesPath, "utf8"));

const TOLERANCE = 1e-9;

/* ------------------------------- normalisasi ------------------------------ */

let normFailures = 0;
let normMaxError = 0;

for (const c of cases.norm) {
  const got = normalizeLandmarks(c.input);
  if (got.length !== c.expected.length) {
    normFailures++;
    continue;
  }
  for (let i = 0; i < got.length; i++) {
    const err = Math.abs(got[i] - c.expected[i]);
    if (err > normMaxError) normMaxError = err;
    if (err > TOLERANCE) normFailures++;
  }
}

/* -------------------------------- prediksi -------------------------------- */

const modelPath = new URL("../web/model.json", import.meta.url);
const model = JSON.parse(readFileSync(modelPath, "utf8"));

let predFailures = 0;
let confMaxError = 0;

for (const c of cases.predict) {
  const r = predict(model, c.features);
  if (!r || r.index !== c.index) {
    predFailures++;
    if (predFailures <= 5) {
      console.log(`  LABEL BEDA: python=${c.label} js=${r ? r.label : "null"}`);
    }
    continue;
  }
  const err = Math.abs(r.confidence - c.confidence);
  if (err > confMaxError) confMaxError = err;
  if (err > 1e-6) predFailures++;
}

/* --------------------------------- laporan -------------------------------- */

console.log(`tipe model       : ${cases.model_type}`);
console.log(
  `normalisasi      : ${cases.norm.length} kasus | selisih maks ${normMaxError.toExponential(2)} | gagal ${normFailures}`,
);
console.log(
  `prediksi         : ${cases.predict.length} kasus | selisih confidence maks ${confMaxError.toExponential(2)} | gagal ${predFailures}`,
);

process.exit(normFailures === 0 && predFailures === 0 ? 0 : 1);
