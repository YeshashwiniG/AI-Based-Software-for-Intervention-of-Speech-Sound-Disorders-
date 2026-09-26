/**
 * PhonoAcoustic AI - Client Application Logic
 * Records/uploads continuous English speech and displays recognized timed words.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global Application State
  const state = {
    audioBlob: null,
    audioUrl: null,
    analysisResult: null,
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    recordStartTime: 0,
    recordTimerInterval: null,
    sentenceId: null,
  };

  // DOM Elements
  const tabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const btnToggleRecord = document.getElementById("btn-toggle-record");
  const micVisualizer = document.getElementById("mic-visualizer");
  const recordStatusText = document.getElementById("record-status-text");
  const recordTimer = document.getElementById("record-timer");
  const audioFileInput = document.getElementById("audio-file-input");
  const btnBrowseFile = document.getElementById("btn-browse-file");
  const fileChosenName = document.getElementById("file-chosen-name");
  const btnRunAnalysis = document.getElementById("btn-run-analysis");
  const sentenceSelect = document.getElementById("sentence-select");
  const sentencePrompt = document.getElementById("sentence-prompt");
  const processingBadge = document.getElementById("processing-badge");

  // Results & Waveform Elements
  const waveformCanvas = document.getElementById("waveform-canvas");
  const canvasCtx = waveformCanvas ? waveformCanvas.getContext("2d") : null;
  const btnPlayAudio = document.getElementById("btn-play-audio");
  const playbackAudio = document.getElementById("playback-audio");
  const playbackTimeLabel = document.getElementById("playback-time-label");
  const audioDurationDisplay = document.getElementById("audio-duration-display");

  // Diagnostic Report Elements
  const diagnosticSummaryBox = document.getElementById("diagnostic-summary-box");
  const summaryIcon = document.getElementById("summary-icon");
  const summaryTitle = document.getElementById("summary-title");
  const summaryDesc = document.getElementById("summary-description");
  const flaggedSection = document.getElementById("flagged-section");
  const flaggedCountBadge = document.getElementById("flagged-count-badge");
  const flaggedList = document.getElementById("flagged-list");
  const wordBreakdownSection = document.getElementById("word-breakdown-section");
  const wordsTimelineChips = document.getElementById("words-timeline-chips");
  const recognizedTranscript = document.getElementById("recognized-transcript");

  // Tab 2 & 3 Elements
  const datasetsGrid = document.getElementById("datasets-cards-grid");
  const targetsGrid = document.getElementById("targets-cards-grid");

  loadSentenceBank();

  async function loadSentenceBank() {
    try {
      const response = await fetch("/api/sentences");
      if (!response.ok) throw new Error("Could not load practice sentences.");
      const data = await response.json();
      sentenceSelect.replaceChildren();
      (data.sentences || []).forEach((sentence, index) => {
        const option = document.createElement("option");
        option.value = sentence.id;
        option.textContent = `Sentence ${index + 1}`;
        option.dataset.text = sentence.text;
        sentenceSelect.appendChild(option);
      });
      state.sentenceId = sentenceSelect.value || null;
      updateSentencePrompt();
    } catch (error) {
      sentencePrompt.textContent = error.message;
      btnRunAnalysis.disabled = true;
    }
  }

  function updateSentencePrompt() {
    const selected = sentenceSelect.selectedOptions[0];
    if (!selected) return;
    state.sentenceId = selected.value;
    sentencePrompt.textContent = `Please say: “${selected.dataset.text}”`;
  }

  sentenceSelect.addEventListener("change", updateSentencePrompt);

  // ==========================================
  // 1. TAB NAVIGATION
  // ==========================================
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // ==========================================
  // 2. LIVE MICROPHONE RECORDING
  // ==========================================
  btnToggleRecord.addEventListener("click", async () => {
    if (!state.isRecording) {
      await startRecording();
    } else {
      stopRecording();
    }
  });

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.audioChunks = [];
      state.mediaRecorder = new MediaRecorder(stream);

      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.audioChunks.push(e.data);
      };

      state.mediaRecorder.onstop = () => {
        createWavBlob(state.audioChunks, state.mediaRecorder.mimeType).then(blob => {
          state.audioBlob = blob;
          state.audioUrl = URL.createObjectURL(blob);
          setAudioForPlayback(state.audioUrl);
          btnRunAnalysis.disabled = false;
          fileChosenName.textContent = `Microphone recording (${(blob.size / 1024).toFixed(1)} KB)`;
        }).catch(err => alert("Could not prepare recording: " + err.message));
      };

      state.mediaRecorder.start();
      state.isRecording = true;
      state.recordStartTime = Date.now();
      micVisualizer.classList.add("recording");
      recordStatusText.textContent = "Recording continuous speech... Click to finish";
      btnRunAnalysis.disabled = true;

      state.recordTimerInterval = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - state.recordStartTime) / 1000);
        const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
        const secs = String(elapsedSec % 60).padStart(2, "0");
        recordTimer.textContent = `${mins}:${secs}`;
      }, 500);

    } catch (err) {
      alert("Microphone access failed: " + err.message);
    }
  }

  function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
      state.mediaRecorder.stop();
      state.mediaRecorder.stream.getTracks().forEach(t => t.stop());
      state.isRecording = false;
      micVisualizer.classList.remove("recording");
      recordStatusText.textContent = "Recording saved. Ready for word-level analysis.";
      clearInterval(state.recordTimerInterval);
    }
  }

  async function createWavBlob(chunks, mimeType) {
    const recorded = new Blob(chunks, { type: mimeType || "audio/webm" });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    try {
      const decoded = await context.decodeAudioData(await recorded.arrayBuffer());
      const sampleRate = decoded.sampleRate;
      const length = decoded.length;
      const channels = Array.from({ length: decoded.numberOfChannels }, (_, i) => decoded.getChannelData(i));
      const mono = new Float32Array(length);
      channels.forEach(channel => {
        for (let i = 0; i < length; i++) mono[i] += channel[i] / channels.length;
      });
      const pcm = new ArrayBuffer(44 + length * 2);
      const view = new DataView(pcm);
      const writeText = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
      writeText(0, "RIFF"); view.setUint32(4, 36 + length * 2, true); writeText(8, "WAVE");
      writeText(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
      view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
      view.setUint16(34, 16, true); writeText(36, "data"); view.setUint32(40, length * 2, true);
      for (let i = 0; i < length; i++) {
        const s = Math.max(-1, Math.min(1, mono[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
      }
      return new Blob([pcm], { type: "audio/wav" });
    } finally {
      await context.close();
    }
  }

  // ==========================================
  // 3. FILE UPLOAD HANDLING
  // ==========================================
  btnBrowseFile.addEventListener("click", () => audioFileInput.click());

  audioFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      state.audioBlob = file;
      state.audioUrl = URL.createObjectURL(file);
      fileChosenName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      setAudioForPlayback(state.audioUrl);
      btnRunAnalysis.disabled = false;
    }
  });

  // ==========================================
  // 4. ANALYSIS EXECUTION
  // ==========================================
  btnRunAnalysis.addEventListener("click", async () => {
    if (state.audioBlob) {
      await executePronunciationAnalysis();
    }
  });

  async function executePronunciationAnalysis() {
    if (!state.sentenceId) {
      alert("Choose a practice sentence first.");
      return;
    }
    setLoadingState(true);
    try {
      const formData = new FormData();
      formData.append("audio_file", state.audioBlob, "utterance.wav");
      formData.append("sentence_id", state.sentenceId);

      const resp = await fetch("/api/analyze-pronunciation", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      state.analysisResult = data;
      renderPronunciation(data);
    } catch (err) {
      alert("Pronunciation analysis failed: " + err.message);
    } finally {
      setLoadingState(false);
    }
  }

  function setLoadingState(isLoading) {
    if (isLoading) {
      btnRunAnalysis.disabled = true;
      processingBadge.textContent = "Analyzing speech...";
      processingBadge.className = "badge warning";
    } else {
      btnRunAnalysis.disabled = false;
      processingBadge.textContent = "Analysis Complete";
      processingBadge.className = "badge success";
    }
  }

  // ==========================================
  // 6. RECOGNIZED WORD RENDERING
  // ==========================================
  function renderPronunciation(data) {
    audioDurationDisplay.textContent = `${data.duration_seconds.toFixed(2)}s`;
    if (data.waveform_envelope) {
      drawWaveformFromEnvelope(data.waveform_envelope);
    }

    const words = data.words || [];
    const analyses = data.word_analysis || [];
    const issues = analyses.filter(item => item.status === "possible_issue");
    const unresolved = analyses.filter(item => item.status === "unresolved" || item.status === "no_supported_sound");
    flaggedSection.style.display = issues.length ? "block" : "none";
    flaggedCountBadge.textContent = `${issues.length} flagged`;
    flaggedList.innerHTML = "";
    issues.forEach(item => {
      const card = document.createElement("article");
      card.className = "flagged-word-card";
      const title = document.createElement("strong");
      title.textContent = `Expected word: ${item.expected_word}`;
      const sounds = document.createElement("p");
      sounds.textContent = `Sound${item.problematic_sounds.length === 1 ? "" : "s"} needing attention: ${item.problematic_sounds.map(sound => `“${sound}”`).join(", ")}`;
      const feedback = document.createElement("p");
      feedback.textContent = "This is a possible practice target based on the model and dataset ratings, not a diagnosis.";
      card.append(title, sounds, feedback);
      flaggedList.appendChild(card);
    });
    const resolvedCount = analyses.length - unresolved.length;
    diagnosticSummaryBox.className = "diagnostic-summary-box " + (issues.length ? "possible-issue" : (resolvedCount ? "speech-recognized" : "empty-state"));
    summaryIcon.textContent = issues.length ? "⚠" : (resolvedCount ? "✓" : "🔍");
    summaryTitle.textContent = issues.length ? "Possible pronunciation issue detected" : (resolvedCount ? "No possible issue detected for supported sounds" : "Sounds could not be analyzed");
    summaryDesc.textContent = `${issues.length} word${issues.length === 1 ? "" : "s"} flagged; ${unresolved.length} word${unresolved.length === 1 ? "" : "s"} unresolved or without a supported sound. This prototype uses whole-word audio for each phone prediction; it does not diagnose.`;
    wordBreakdownSection.style.display = "block";
    recognizedTranscript.textContent = data.transcript || "";
    wordsTimelineChips.innerHTML = "";
    analyses.forEach(item => {
      const chip = document.createElement("div");
      chip.className = "word-chip" + (item.status === "possible_issue" ? " possible-issue-chip" : (item.status === "unresolved" ? " unresolved-chip" : ""));
      const word = document.createElement("span");
      word.className = "chip-word-text";
      word.textContent = item.expected_word;
      const timing = document.createElement("span");
      timing.className = "chip-timing";
      timing.textContent = item.start == null ? "not aligned" : `${Number(item.start).toFixed(2)}–${Number(item.end).toFixed(2)}s`;
      const soundInfo = document.createElement("span");
      soundInfo.className = "chip-timing";
      soundInfo.textContent = item.sound_results?.length
        ? item.sound_results.map(sound => `${sound.sound}: ${sound.status === "possible_issue" ? "may need attention" : "no issue detected"}`).join(" · ")
        : (item.status === "no_supported_sound" ? "no validated target sound" : "sound analysis unavailable");
      chip.append(word, timing, soundInfo);
      wordsTimelineChips.appendChild(chip);
    });
    if (!analyses.length) wordsTimelineChips.textContent = words.length
      ? "No expected words could be aligned to recognized audio."
      : "No words recognized. Try recording the complete sentence again.";
  }

  // ==========================================
  // 7. WAVEFORM & AUDIO PLAYBACK
  // ==========================================
  function setAudioForPlayback(url) {
    playbackAudio.src = url;
    btnPlayAudio.disabled = false;
    playbackAudio.onloadedmetadata = () => {
      audioDurationDisplay.textContent = `${playbackAudio.duration.toFixed(2)}s`;
      updatePlaybackTimeDisplay();
    };
    playbackAudio.ontimeupdate = updatePlaybackTimeDisplay;
    playbackAudio.onended = () => {
      btnPlayAudio.querySelector("span").textContent = "Play Utterance";
    };
  }

  btnPlayAudio.addEventListener("click", () => {
    if (playbackAudio.paused) {
      playbackAudio.play();
      btnPlayAudio.querySelector("span").textContent = "Pause";
    } else {
      playbackAudio.pause();
      btnPlayAudio.querySelector("span").textContent = "Play Utterance";
    }
  });

  function updatePlaybackTimeDisplay() {
    const cur = playbackAudio.currentTime || 0;
    const dur = playbackAudio.duration || 0;
    playbackTimeLabel.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
  }

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  function drawWaveformFromEnvelope(envelope) {
    if (!canvasCtx || !waveformCanvas || !envelope) return;
    const w = waveformCanvas.width;
    const h = waveformCanvas.height;
    canvasCtx.clearRect(0, 0, w, h);

    const barWidth = w / envelope.length;
    for (let i = 0; i < envelope.length; i++) {
      const val = envelope[i];
      const bh = Math.max(4, val * (h - 10));
      
      // Gradient color based on intensity
      const grad = canvasCtx.createLinearGradient(0, (h - bh) / 2, 0, (h + bh) / 2);
      grad.addColorStop(0, "hsl(194, 98%, 56%)");
      grad.addColorStop(1, "hsl(242, 88%, 68%)");
      
      canvasCtx.fillStyle = grad;
      canvasCtx.fillRect(i * barWidth, (h - bh) / 2, Math.max(1, barWidth - 1.5), bh);
    }
  }

  // ==========================================
  // 8. DATASETS & TARGET SOUNDS INITIALIZATION
  // ==========================================
  async function loadDatasets() {
    try {
      const resp = await fetch("/api/datasets");
      const data = await resp.json();
      datasetsGrid.innerHTML = "";

      datasetsGrid.innerHTML = "";
      data.datasets.forEach(ds => {
        const card = document.createElement("div");
        card.className = "dataset-card";
        card.innerHTML = `<div class="ds-header"><h3>${escapeHtml(ds.name)}</h3></div>
          <p>Participants: ${ds.participants} · Recordings: ${ds.recordings} · Usable: ${ds.usable_recordings}</p>
          <p>${escapeHtml(ds.labels)}</p><p>Source: ${escapeHtml(ds.source)}</p>`;
        datasetsGrid.appendChild(card);
      });
      if (!data.datasets.length) datasetsGrid.textContent = data.message || "Dataset report is not available yet.";
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  }

  async function loadTargetSounds() {
    try {
      const resp = await fetch("/api/target-sounds");
      const data = await resp.json();
      targetsGrid.innerHTML = "";

      Object.entries(data.targets).forEach(([phone, count]) => {
        const card = document.createElement("div");
        card.className = "target-sound-card";
        card.innerHTML = `<div class="tsc-header"><span class="tsc-ipa">/${escapeHtml(phone)}/</span></div>
          <div class="tsc-body"><h4>${count.toLocaleString()} annotated examples</h4>
          <p>Annotation count only. The current model does not assess or locate this sound in a user's recording.</p></div>`;
        targetsGrid.appendChild(card);
      });
      if (!Object.keys(data.targets).length) targetsGrid.textContent = data.message || "No target phone counts available.";
    } catch (err) {
      console.error("Failed to load target sounds:", err);
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial Load
  loadDatasets();
  loadTargetSounds();
});
