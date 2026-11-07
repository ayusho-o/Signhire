/**
 * web/static/app.js
 * -----------------
 * SignHire front-end logic.
 *
 * Responsibilities
 * ----------------
 * 1. Poll /status every 300 ms and update the prediction HUD.
 * 2. Manage the Web Speech API for the interviewer's microphone.
 * 3. Send committed signs (applicant) and speech (interviewer) to the
 *    Flask backend and update the shared transcript.
 * 4. Poll /transcript every 1 s to keep the transcript in sync.
 * 5. Load /config on startup to populate the language dropdown.
 *
 * All fetch calls use try/catch so a backend error never crashes the UI.
 */

"use strict";

/* ── DOM references ────────────────────────────────────────────────── */
const predLabel      = document.getElementById("prediction-label");
const predBox        = document.getElementById("prediction-box");
const confFill       = document.getElementById("confidence-fill");
const confPct        = document.getElementById("confidence-pct");
const threshMarker   = document.getElementById("threshold-marker");
const bufferBar      = document.getElementById("buffer-bar");
const sendSignBtn    = document.getElementById("send-sign-btn");
const langSelect     = document.getElementById("lang-select");
const micBtn         = document.getElementById("mic-btn");
const speechStatus   = document.getElementById("speech-status");
const interimText    = document.getElementById("interim-text");
const transcriptBox  = document.getElementById("transcript-box");
const clearBtn       = document.getElementById("clear-btn");

/* ── State ─────────────────────────────────────────────────────────── */
let currentResult    = null;   // latest /status response
let confidenceThresh = 0.70;   // loaded from /config
let isRecognising    = false;
let recognition      = null;

/* ── 1. Load config ────────────────────────────────────────────────── */
async function loadConfig() {
  try {
    const resp = await fetch("/config");
    const cfg  = await resp.json();

    confidenceThresh = cfg.threshold || 0.70;

    // Populate language dropdown
    const existingCodes = Array.from(langSelect.options)
                               .map(o => o.value);
    (cfg.target_languages || []).forEach(lang => {
      if (existingCodes.includes(lang.code)) {
        // Update disabled state on existing option
        const opt = langSelect.querySelector(`option[value="${lang.code}"]`);
        if (opt && lang.disabled) {
          opt.disabled = true;
          opt.textContent = `${lang.label} (translator not configured)`;
        }
        return;
      }
      const opt = document.createElement("option");
      opt.value    = lang.code;
      opt.textContent = lang.disabled
        ? `${lang.label} (translator not configured)`
        : lang.label;
      opt.disabled = !!lang.disabled;
      langSelect.appendChild(opt);
    });

    // Set threshold marker position
    threshMarker.style.left = `${confidenceThresh * 100}%`;

  } catch (e) {
    console.warn("Could not load /config:", e);
  }
}

/* ── 2. Status polling ─────────────────────────────────────────────── */
async function pollStatus() {
  try {
    const resp = await fetch("/status");
    if (!resp.ok) return;
    const data = await resp.json();
    currentResult = data;
    updatePredictionHUD(data);
  } catch (e) {
    /* backend not ready yet — ignore */
  }
}

function updatePredictionHUD(data) {
  const label      = data.label      || "Waiting\u2026";
  const confidence = data.confidence || 0.0;
  const isUnsure   = data.is_unsure  !== false ? true : data.is_unsure;
  const bufFill    = data.buffer_fill || 0.0;

  // Label
  predLabel.textContent = label;

  // Prediction box class
  predBox.classList.toggle("confident", !isUnsure);
  predBox.classList.toggle("unsure",     isUnsure);

  // Confidence bar
  const pct = Math.round(confidence * 100);
  confFill.style.width      = `${pct}%`;
  confFill.style.background = isUnsure ? "#e67e22" : "#2ecc71";
  confPct.textContent       = `${pct}%`;

  // Buffer bar (CSS custom property on the pseudo-element)
  bufferBar.style.setProperty("--fill", `${Math.round(bufFill * 100)}%`);

  // Enable send button only when we have a confident prediction
  const canSend = !isUnsure &&
                  label !== "Waiting\u2026" &&
                  label !== "";
  sendSignBtn.disabled = !canSend;
}

/* ── 3. Send sign to transcript ────────────────────────────────────── */
sendSignBtn.addEventListener("click", async () => {
  if (!currentResult || currentResult.is_unsure) return;

  const targetLang = langSelect.value || "en";

  try {
    const resp = await fetch("/applicant_sign", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ target_lang: targetLang }),
    });
    const data = await resp.json();
    if (data.ok) {
      appendTranscriptEntry(data.entry);
    } else {
      console.warn("Could not commit sign:", data.error);
    }
  } catch (e) {
    console.error("Send sign failed:", e);
  }
});

/* ── 4. Web Speech API ─────────────────────────────────────────────── */
function initSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    micBtn.disabled     = true;
    micBtn.textContent  = "🎤 Speech API not supported (use Chrome/Edge)";
    speechStatus.textContent =
      "Web Speech API is not available in this browser.";
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous      = true;
  recognition.interimResults  = true;
  recognition.lang            = "en-IN";  // Indian English

  recognition.onstart = () => {
    isRecognising = true;
    micBtn.textContent = "⏹ Stop listening";
    micBtn.classList.add("recording");
    speechStatus.textContent = "Listening…";
    interimText.textContent  = "";
  };

  recognition.onend = () => {
    isRecognising = false;
    micBtn.textContent = "🎤 Start speaking";
    micBtn.classList.remove("recording");
    speechStatus.textContent = "Stopped.";
    interimText.textContent  = "";
  };

  recognition.onerror = (event) => {
    speechStatus.textContent = `Error: ${event.error}`;
    isRecognising = false;
    micBtn.textContent = "🎤 Start speaking";
    micBtn.classList.remove("recording");
  };

  recognition.onresult = (event) => {
    let interimTranscript = "";
    let finalTranscript   = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript;
      } else {
        interimTranscript += transcript;
      }
    }

    interimText.textContent = interimTranscript;

    if (finalTranscript.trim()) {
      sendSpeechToBackend(finalTranscript.trim());
      interimText.textContent = "";
    }
  };
}

micBtn.addEventListener("click", () => {
  if (!recognition) return;
  if (isRecognising) {
    recognition.stop();
  } else {
    recognition.start();
  }
});

async function sendSpeechToBackend(text) {
  try {
    const resp = await fetch("/interviewer_speech", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });
    const data = await resp.json();
    if (data.ok) {
      appendTranscriptEntry(data.entry);
    }
  } catch (e) {
    console.error("Speech send failed:", e);
  }
}

/* ── 5. Transcript ─────────────────────────────────────────────────── */
function appendTranscriptEntry(entry) {
  // Remove "no messages yet" placeholder
  const empty = transcriptBox.querySelector(".transcript-empty");
  if (empty) empty.remove();

  const div = document.createElement("div");
  div.className = `transcript-entry ${entry.speaker}`;
  div.dataset.time = entry.time;

  const meta = document.createElement("div");
  meta.className = "entry-meta";

  const speaker = document.createElement("span");
  speaker.className   = "entry-speaker";
  speaker.textContent = entry.speaker;

  const ts = document.createElement("span");
  ts.className   = "entry-time";
  ts.textContent = entry.time;

  meta.appendChild(speaker);
  meta.appendChild(ts);

  const textEl = document.createElement("div");
  textEl.className   = "entry-text";
  textEl.textContent = entry.text;

  div.appendChild(meta);
  div.appendChild(textEl);

  // Show translation if available and different from original
  if (entry.translated && entry.translated !== entry.text) {
    const trans = document.createElement("div");
    trans.className   = "entry-translated";
    trans.textContent = `→ ${entry.translated}`;
    div.appendChild(trans);
  }

  transcriptBox.appendChild(div);
  // Auto-scroll to latest message
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

async function refreshTranscript() {
  try {
    const resp = await fetch("/transcript");
    const data = await resp.json();
    const entries = data.transcript || [];

    // Only redraw if count changed (avoids flicker)
    const current = transcriptBox.querySelectorAll(".transcript-entry").length;
    if (entries.length === current) return;

    // Full redraw
    transcriptBox.innerHTML = "";
    if (entries.length === 0) {
      transcriptBox.innerHTML =
        '<p class="transcript-empty">No messages yet.</p>';
      return;
    }
    entries.forEach(appendTranscriptEntry);
  } catch (e) { /* ignore */ }
}

clearBtn.addEventListener("click", async () => {
  try {
    await fetch("/clear_transcript", { method: "POST" });
    transcriptBox.innerHTML =
      '<p class="transcript-empty">No messages yet.</p>';
  } catch (e) { console.error("Clear failed:", e); }
});

/* ── Startup ───────────────────────────────────────────────────────── */
(async function init() {
  await loadConfig();
  initSpeechRecognition();

  // Poll status every 300 ms
  setInterval(pollStatus, 300);

  // Poll transcript every 1 s
  setInterval(refreshTranscript, 1000);

  // Initial status fetch
  pollStatus();
  refreshTranscript();
})();
