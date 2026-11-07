# SignHire — Real-Time ISL Interview Communication Aid

A university prototype that enables two-way communication between a deaf applicant
who uses Indian Sign Language (ISL) and a hearing interviewer.

> **Dataset notice:** All results in this project come from a small self-recorded
> dataset (1 signer, 8 signs, 5 clips per sign — 40 clips total). They are
> indicative only and do not generalise to other signers or conditions.
> See [reports/CAVEAT.md](reports/CAVEAT.md) for details.

---

## How it works

| Who | What they do | What the system does |
|-----|-------------|----------------------|
| Applicant (deaf) | Signs in front of webcam | MediaPipe extracts landmarks → LSTM/GRU model predicts sign → shown on screen with confidence score |
| Interviewer (hearing) | Speaks into microphone | Browser Web Speech API converts speech to text → shown to applicant |
| Both | — | Shared transcript of every exchange is shown on screen |

---

## Project structure

```
SignHire/
├── config/
│   ├── vocabulary.yaml      # sign labels (edit to add/remove signs)
│   └── settings.yaml        # all tunable numbers
├── data/
│   ├── raw/                 # recorded .npy landmark files + .json metadata
│   ├── processed/           # resampled dataset (X.npy, y.npy, meta.json)
│   └── models/              # trained .keras model files
├── src/
│   ├── capture/             # landmark_viewer.py, live_predictor.py
│   ├── recorder/            # recorder.py — guided clip recorder
│   ├── landmarks/           # extractor.py — MediaPipe wrapper
│   ├── preprocess/          # preprocess.py — resampling + augmentation
│   ├── models/              # lstm_model.py, baseline_model.py
│   ├── train/               # train.py — cross-validation training loop
│   ├── evaluate/            # evaluate.py — metrics + plots
│   ├── inference/           # inference.py — real-time engine
│   ├── translate/           # translator.py — pluggable translation
│   └── app/                 # app.py — Flask web server
├── web/
│   ├── index.html           # combined UI
│   └── static/              # style.css, app.js
├── reports/                 # confusion matrices, training curves, metrics
├── tests/                   # unit tests
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements
- Python **3.10 or 3.11** (mediapipe does not support 3.12+)
- Windows / macOS / Linux
- A webcam

### Install

```powershell
# 1 — navigate to the project folder
cd "C:\Users\Aradhy Bhardwaj\OneDrive\Desktop\Forma-AI\SignHire"

# 2 — create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

# 3 — install dependencies
pip install -r requirements.txt
```

---

## Running the project

### Step 1 — Check your camera index

```powershell
python find_camera.py
```

Note the index of your built-in camera (usually 0 or 1). Use it in all commands below.

### Step 2 — View the landmark overlay (optional sanity check)

```powershell
python src/capture/landmark_viewer.py
```

Press **Q** to quit. You should see your webcam feed with the MediaPipe
pose skeleton, hand skeletons, and face landmark dots.

### Step 3 — Record your own signs

```powershell
python src/recorder/recorder.py --signer signer_01 --camera 1
```

- Press **SPACE** to start each clip
- Sign the word shown on screen during the red-border recording
- Press **K** to keep, **R** to re-record
- The session saves automatically; press **Q** to resume later

Target: 8 signs × 5+ clips per signer. More signers = better evaluation.

### Step 4 — Build the processed dataset

```powershell
python src/preprocess/preprocess.py
```

### Step 5 — Train both models

```powershell
python src/train/train.py
```

This runs stratified 5-fold CV (or LOSO if you have ≥2 signers) for both
the LSTM and GRU baseline. Training curves and confusion matrices are saved
to `reports/`.

### Step 6 — Live prediction (standalone test)

```powershell
python src/capture/live_predictor.py --camera 1
```

### Step 7 — Run the full web interface

```powershell
python src/app/app.py --camera 1
```

Open **http://localhost:5000** in **Chrome or Edge**.

---

## Demo walkthrough

1. Open `http://localhost:5000` in Chrome or Edge.
2. **Applicant side (left panel):**
   - Your webcam feed with the skeleton overlay appears.
   - The buffer bar fills as 30 frames are collected.
   - Once full, the predicted sign appears in large text with a confidence bar.
   - When confidence ≥ 70% the label turns green; below 70% shows orange "Unsure, please repeat".
   - Click **✋ Send this sign** to add the prediction to the shared transcript.
3. **Interviewer side (right panel):**
   - Click **🎤 Start speaking** and speak a sentence.
   - Your words appear in the transcript automatically.
   - Click again to stop.
4. **Shared transcript** (bottom): every committed sign and every spoken sentence appears here with timestamps.

---

## Running the tests

```powershell
pytest tests/ -v
```

Tests cover: landmark extractor, resampling, augmentation, LOSO splits,
and inference engine.

---

## Configuration

All tunable parameters are in `config/settings.yaml`. Key ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recording.sequence_frames` | 30 | Frames per clip |
| `recording.fps` | 15 | Camera capture fps |
| `recording.clips_per_sign` | 5 | Target clips per sign |
| `augmentation.copies_per_clip` | 4 | Augmented copies per training clip |
| `inference.confidence_threshold` | 0.70 | Below this → "Unsure" |
| `inference.predict_every_n_frames` | 5 | Inference frequency |
| `model.epochs` | 60 | Max training epochs |

To add a sign: add its name (quoted) to `config/vocabulary.yaml`, record clips, retrain.

---

## Translation

By default, translation is disabled (English only). To enable Hindi translation:

1. Set up a [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) instance
   (self-hosted or use a public endpoint).
2. Update `config/settings.yaml`:
   ```yaml
   translation:
     provider: libretranslate
     libretranslate_url: https://your-instance.example.com
   ```
3. Set your API key as an environment variable:
   ```powershell
   $env:LIBRETRANSLATE_API_KEY = "your-key-here"
   ```

---

## Known limitations

See the full list in [LIMITATIONS.md](LIMITATIONS.md).

Short version:
- Small dataset: 1 signer, 8 signs, 5 clips each. Not representative.
- No signer-independent evaluation yet (needs ≥2 signers for LOSO).
- Web Speech API requires Chrome/Edge and an internet connection.
- Inference latency ~67 ms/sample on CPU — acceptable but not instant.
- Signs must be performed clearly and in isolation (no continuous signing).
- Hindi translation requires external service configuration.
- All data was recorded under consistent lighting; performance may degrade in other conditions.

---

## Tech stack

| Component | Library | License |
|-----------|---------|---------|
| Landmark extraction | MediaPipe 0.10.14 | Apache 2.0 |
| Video capture | OpenCV 4.9 | Apache 2.0 |
| Model training | TensorFlow/Keras 2.15 | Apache 2.0 |
| Numerics | NumPy 1.26, SciPy 1.13 | BSD |
| Metrics | scikit-learn 1.4 | BSD |
| Plots | Matplotlib 3.8 | PSF/BSD |
| Web server | Flask 3.0 | BSD |
| Translation | requests 2.31 (HTTP client) | Apache 2.0 |
| Speech-to-text | Browser Web Speech API | Browser built-in |
| Config | PyYAML 6.0 | MIT |

No code was copied from other repositories. All original.
