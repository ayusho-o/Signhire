# SignHire — Known Limitations

This document is part of the honest evaluation required for a university
prototype. Every limitation listed here is real and should be discussed
when presenting the project.

---

## 1. Very small dataset

- **What we have:** 1 signer, 8 signs, 5 clips each = 40 clips total.
- **Why it matters:** Machine learning models need diverse data to generalise.
  With one signer, the model has only seen one person's signing style,
  hand size, and movement range.
- **What it means for accuracy:** The reported accuracy (GRU baseline: 0.825,
  LSTM: 0.725) reflects performance on held-out clips from the *same* signer
  under similar conditions. It does not tell us how the model would perform
  for a different person.
- **How to improve:** Record at least 3 signers × 20 clips each and use
  leave-one-signer-out evaluation.

## 2. No signer-independent evaluation

- **What we did:** Stratified 5-fold cross-validation on a single signer's
  data. This is labelled clearly in all reports.
- **What we did NOT do:** Leave-one-signer-out (LOSO) — this requires ≥2
  signers. The code supports LOSO automatically when more signers are added.
- **Why it matters:** LOSO is the only evaluation that tests whether the
  model generalises to new people. Without it, reported accuracy is
  optimistic.

## 3. Isolated signs only

- The model recognises one sign at a time from a 30-frame (≈2 second) window.
- It cannot handle continuous signing, signs that flow into each other, or
  sentences.
- The applicant must pause between signs and manually press "Send this sign"
  to commit a prediction.

## 4. Fixed vocabulary of 8 signs

- Only the 8 signs in `config/vocabulary.yaml` can be recognised.
- Any sign outside this set will either be misclassified as one of the 8
  or flagged as "Unsure".
- Extending the vocabulary requires recording new data and retraining.

## 5. Lighting and environment sensitivity

- All clips were recorded in the same room under similar lighting.
- Performance may degrade significantly in poor lighting, with a cluttered
  background, or with the signer far from the camera.
- MediaPipe landmark detection fails silently (returns zeros) when hands or
  pose are not detected — the model then receives a zero vector, which
  typically triggers the "Unsure" gate.

## 6. Web Speech API limitations

- Speech recognition uses the browser's built-in Web Speech API.
- This requires **Chrome or Edge** — it does not work in Firefox or Safari.
- It requires an **internet connection** (Google's speech service is used
  under the hood).
- It performs best with clear speech and a quiet environment.
- Language is set to `en-IN` (Indian English) which may not recognise all
  accents well.

## 7. Inference latency

- Mean inference latency is ~67 ms/sample on a laptop CPU.
- This is acceptable for real-time use (predictions update every 5 frames),
  but there is a noticeable delay between signing and seeing the prediction.
- A GPU would reduce this significantly, but the project is designed for
  CPU-only use.

## 8. Translation not functional by default

- Hindi translation requires an external LibreTranslate instance.
- By default, the translation provider is set to `none` and Hindi shows
  "Translator not configured".
- This is by design — we do not claim translation works unless it is
  explicitly configured and tested.

## 9. No authentication or access control

- The Flask app runs locally with no login or session management.
- It is designed for single-machine local use only and should not be
  deployed on a public network.

## 10. Model overfitting risk

- With 40 training clips (expanded to ~160 with augmentation), even a
  small model can memorise patterns.
- The reported validation accuracy of 1.000 in many folds is a sign of
  this — the model is fitting the training distribution closely.
- Test accuracy (0.625–0.875) is the honest number to report.

---

## What would make this production-ready

1. At least 20 signers, 50+ clips per sign per signer.
2. Leave-one-signer-out evaluation with published results.
3. Continuous signing recognition (not isolated words).
4. A larger, publicly available ISL dataset for pre-training.
5. Proper deployment infrastructure (HTTPS, authentication, load testing).
6. Formal accessibility audit with deaf community involvement.
