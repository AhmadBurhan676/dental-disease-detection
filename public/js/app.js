/**
 * DentalVision AI - Panoramic Radiography Pathology Diagnosis
 * Frontend Controller & Visualization Engine
 */

document.addEventListener("DOMContentLoaded", () => {

  // ── Theme Toggle (Light / Dark) ──────────────────────────────────────────
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const savedTheme = localStorage.getItem("dental_theme") || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", savedTheme);
  applyThemeIcon(savedTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("dental_theme", next);
      applyThemeIcon(next);
    });
  }

  function applyThemeIcon(theme) {
    if (!themeToggleBtn) return;
    if (theme === "dark") {
      themeToggleBtn.innerHTML = `<i class="fa-solid fa-sun fa-spin" style="color:#fbbf24;"></i>`;
      themeToggleBtn.title = "Switch to Light Mode";
    } else {
      themeToggleBtn.innerHTML = `<i class="fa-solid fa-moon fa-beat" style="color:#818cf8;"></i>`;
      themeToggleBtn.title = "Switch to Dark Mode";
    }
  }
  // ────────────────────────────────────────────────────────────────────────

  // DOM Elements
  const tabBtns = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const imageWrapper = document.getElementById("imageWrapper");
  const activeRadiograph = document.getElementById("activeRadiograph");
  const scanLine = document.getElementById("scanLine");
  const inferenceOverlay = document.getElementById("inferenceOverlay");

  const camSelectorBar = document.getElementById("camSelectorBar");
  const camTabs = document.querySelectorAll(".cam-tab");
  const opacitySlider = document.getElementById("opacitySlider");
  const opacityVal = document.getElementById("opacityVal");

  const btnEnhance = document.getElementById("btnEnhance");
  const btnInvert = document.getElementById("btnInvert");
  const btnResetView = document.getElementById("btnResetView");
  const btnNewScan = document.getElementById("btnNewScan");
  const btnPrintReport = document.getElementById("btnPrintReport");

  const samplesContainer = document.getElementById("samplesContainer");

  // Summary Elements
  const statusRing = document.getElementById("statusRing");
  const statusTitle = document.getElementById("statusTitle");
  const statusSummary = document.getElementById("statusSummary");
  const lblLatency = document.getElementById("lblLatency");
  const lblPatientId = document.getElementById("lblPatientId");

  // Disease 1: Caries
  const probCaries = document.getElementById("probCaries");
  const barCaries = document.getElementById("barCaries");
  const badgeCaries = document.getElementById("badgeCaries");
  const severityCaries = document.getElementById("severityCaries");
  const treatmentCaries = document.getElementById("treatmentCaries");

  // Disease 2: Periapical
  const probPeriapical = document.getElementById("probPeriapical");
  const barPeriapical = document.getElementById("barPeriapical");
  const badgePeriapical = document.getElementById("badgePeriapical");
  const severityPeriapical = document.getElementById("severityPeriapical");
  const treatmentPeriapical = document.getElementById("treatmentPeriapical");

  // Disease 3: Periodontitis
  const probPeriodontitis = document.getElementById("probPeriodontitis");
  const barPeriodontitis = document.getElementById("barPeriodontitis");
  const badgePeriodontitis = document.getElementById("badgePeriodontitis");
  const severityPeriodontitis = document.getElementById("severityPeriodontitis");
  const treatmentPeriodontitis = document.getElementById("treatmentPeriodontitis");

  // GPU & Training Elements
  const gpuModel = document.getElementById("gpuModel");
  const trainGpuName = document.getElementById("trainGpuName");
  const trainVram = document.getElementById("trainVram");
  const trainCudaVer = document.getElementById("trainCudaVer");
  const btnStartTraining = document.getElementById("btnStartTraining");
  const epochInput = document.getElementById("epochInput");
  const batchInput = document.getElementById("batchInput");
  const lrInput = document.getElementById("lrInput");
  const trainingProgressBox = document.getElementById("trainingProgressBox");
  const trainingProgressBar = document.getElementById("trainingProgressBar");
  const trainingPercent = document.getElementById("trainingPercent");
  const trainingStatusText = document.getElementById("trainingStatusText");
  const trainingTerminal = document.getElementById("trainingTerminal");
  const trainingPlotImg = document.getElementById("trainingPlotImg");

  // State Variables
  let currentPredictionData = null;
  let currentLayer = "composite_heatmap";
  let isInverted = false;
  let isEnhanceActive = true;
  let currentFile = null;
  let trainingPollInterval = null;

  // 1. Initialize
  initHardwareAndModelInfo();
  loadSampleRadiographs();
  setupTabs();
  setupDropzone();
  setupViewTools();
  setupCamSelector();
  setupTraining();

  // Navigation Tabs
  function setupTabs() {
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        tabBtns.forEach(b => b.classList.remove("active"));
        tabPanes.forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        const target = btn.dataset.tab;
        const targetPane = document.getElementById(target);
        if (targetPane) targetPane.classList.add("active");
      });
    });
  }

  // Fetch Model & GPU Specs
  async function initHardwareAndModelInfo() {
    try {
      const res = await fetch("/api/model-info");
      const data = await res.json();
      if (data && data.hardware) {
        const hw = data.hardware;
        if (hw.available) {
          gpuModel.textContent = hw.device;
          trainGpuName.textContent = hw.device;
          trainVram.textContent = `${hw.vram_total_gb} GB`;
          trainCudaVer.textContent = `CUDA ${hw.cuda_version}`;
        }
      }
    } catch (err) {
      console.warn("Model info fetch error:", err);
    }
  }

  // Load Authentic Samples
  async function loadSampleRadiographs() {
    try {
      const res = await fetch("/api/samples");
      const samples = await res.json();

      samplesContainer.innerHTML = "";
      samples.forEach(sample => {
        const item = document.createElement("div");
        item.className = "sample-thumbnail";
        item.innerHTML = `
          <img src="${sample.url}" alt="${sample.title}" loading="lazy">
          <div class="sample-meta">
            <span class="sample-title">${sample.title}</span>
            <span class="sample-conditions">${sample.conditions ? sample.conditions.join(", ") : "OPG"}</span>
          </div>
        `;
        item.addEventListener("click", () => loadSampleFromUrl(sample.url, sample.title));
        samplesContainer.appendChild(item);
      });
    } catch (err) {
      console.warn("Could not load samples:", err);
    }
  }

  // Load Sample Radiograph
  async function loadSampleFromUrl(url, title) {
    showInferenceLoading(true);
    try {
      const blob = await fetch(url).then(r => r.blob());
      const file = new File([blob], title.replace(/\s+/g, "_") + ".jpg", { type: "image/jpeg" });
      currentFile = file;
      await runDiagnosis(file);
    } catch (err) {
      alert("Error loading sample image: " + err.message);
      showInferenceLoading(false);
    }
  }

  // Dropzone & File Input
  function setupDropzone() {
    ["dragenter", "dragover"].forEach(eventName => {
      dropzone.addEventListener(eventName, e => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--accent-cyan)";
        dropzone.style.background = "rgba(6, 182, 212, 0.08)";
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropzone.addEventListener(eventName, e => {
        e.preventDefault();
        dropzone.style.borderColor = "rgba(6, 182, 212, 0.3)";
        dropzone.style.background = "rgba(6, 182, 212, 0.02)";
      });
    });

    dropzone.addEventListener("drop", e => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFileSelect(files[0]);
      }
    });

    fileInput.addEventListener("change", e => {
      if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });

    btnNewScan.addEventListener("click", () => {
      dropzone.style.display = "flex";
      imageWrapper.style.display = "none";
      camSelectorBar.style.display = "none";
      fileInput.value = "";
      currentPredictionData = null;
      resetResults();
    });
  }

  function handleFileSelect(file) {
    if (!file.type.match("image.*")) {
      alert("Please upload a panoramic X-ray image (JPEG, PNG, TIFF).");
      return;
    }
    currentFile = file;
    runDiagnosis(file);
  }

  // Run Diagnosis Inference
  async function runDiagnosis(file) {
    showInferenceLoading(true);
    generateRandomPatientId();

    const formData = new FormData();
    formData.append("file", file);
    formData.append("enhance", isEnhanceActive);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Inference failed.");
      }

      const data = await res.json();
      currentPredictionData = data;
      renderDiagnosisResults(data);

      // Show radiograph with initial heatmap
      dropzone.style.display = "none";
      imageWrapper.style.display = "flex";
      camSelectorBar.style.display = "flex";
      updateActiveLayer();

    } catch (err) {
      alert("Diagnosis failed: " + err.message);
    } finally {
      showInferenceLoading(false);
    }
  }

  // Render Predictions to UI
  function renderDiagnosisResults(data) {
    const report = data.report;
    const diseases = report.diseases;

    // Latency
    lblLatency.innerHTML = `<i class="fa-solid fa-bolt"></i> ${report.hardware_accelerator}: <strong>${report.inference_time_ms} ms</strong>`;

    // Overall Status
    statusTitle.textContent = report.overall_health;
    statusSummary.textContent = report.summary;
    statusRing.style.borderColor = report.health_color;
    statusRing.style.color = report.health_color;
    statusRing.style.background = `${report.health_color}1a`;

    // 1. Dental Caries (Index 0)
    const caries = diseases[0];
    probCaries.textContent = `${caries.probability}%`;
    barCaries.style.width = `${caries.probability}%`;
    badgeCaries.textContent = caries.badge;
    badgeCaries.className = `badge badge-${caries.badge.toLowerCase()}`;
    severityCaries.textContent = caries.severity;
    treatmentCaries.textContent = caries.clinical_guidance;

    // 2. Periapical Lesion (Index 1)
    const periapical = diseases[1];
    probPeriapical.textContent = `${periapical.probability}%`;
    barPeriapical.style.width = `${periapical.probability}%`;
    badgePeriapical.textContent = periapical.badge;
    badgePeriapical.className = `badge badge-${periapical.badge.toLowerCase()}`;
    severityPeriapical.textContent = periapical.severity;
    treatmentPeriapical.textContent = periapical.clinical_guidance;

    // 3. Periodontitis (Index 2)
    const periodontitis = diseases[2];
    probPeriodontitis.textContent = `${periodontitis.probability}%`;
    barPeriodontitis.style.width = `${periodontitis.probability}%`;
    badgePeriodontitis.textContent = periodontitis.badge;
    badgePeriodontitis.className = `badge badge-${periodontitis.badge.toLowerCase()}`;
    severityPeriodontitis.textContent = periodontitis.severity;
    treatmentPeriodontitis.textContent = periodontitis.clinical_guidance;
  }

  // Reset Results
  function resetResults() {
    statusTitle.textContent = "Awaiting Radiograph Upload";
    statusSummary.textContent = "Select a sample or upload an OPG radiograph to view deep learning disease classification.";
    statusRing.style.borderColor = "var(--accent-emerald)";
    statusRing.style.color = "var(--accent-emerald)";
    statusRing.style.background = "rgba(16, 185, 129, 0.1)";
    lblLatency.innerHTML = `<i class="fa-solid fa-stopwatch"></i> GPU: -- ms`;

    [barCaries, barPeriapical, barPeriodontitis].forEach(b => b.style.width = "0%");
    [probCaries, probPeriapical, probPeriodontitis].forEach(p => p.textContent = "0.0%");
    [badgeCaries, badgePeriapical, badgePeriodontitis].forEach(b => {
      b.textContent = "PENDING";
      b.className = "badge badge-clear";
    });
    [severityCaries, severityPeriapical, severityPeriodontitis].forEach(s => s.textContent = "--");
    [treatmentCaries, treatmentPeriapical, treatmentPeriodontitis].forEach(t => t.textContent = "--");
  }

  // Viewport Tools: CLAHE, Invert, Reset
  function setupViewTools() {
    btnEnhance.classList.add("active");

    btnEnhance.addEventListener("click", () => {
      isEnhanceActive = !isEnhanceActive;
      btnEnhance.classList.toggle("active", isEnhanceActive);
      if (currentFile) {
        runDiagnosis(currentFile);
      }
    });

    btnInvert.addEventListener("click", () => {
      isInverted = !isInverted;
      btnInvert.classList.toggle("active", isInverted);
      activeRadiograph.style.filter = isInverted ? "invert(1) contrast(1.15)" : "none";
    });

    btnResetView.addEventListener("click", () => {
      isInverted = false;
      btnInvert.classList.remove("active");
      activeRadiograph.style.filter = "none";
      opacitySlider.value = 75;
      opacityVal.textContent = "75%";
      activeRadiograph.style.opacity = "1";
    });

    btnPrintReport.addEventListener("click", () => {
      window.print();
    });
  }

  // Grad-CAM Attention Layer Switcher
  function setupCamSelector() {
    camTabs.forEach(tab => {
      tab.addEventListener("click", () => {
        camTabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        currentLayer = tab.dataset.layer;
        updateActiveLayer();
      });
    });

    opacitySlider.addEventListener("input", e => {
      const val = e.target.value;
      opacityVal.textContent = `${val}%`;
      activeRadiograph.style.opacity = (val / 100).toString();
    });
  }

  function updateActiveLayer() {
    if (!currentPredictionData || !currentPredictionData.visualizations) return;
    const viz = currentPredictionData.visualizations;
    const imgSrc = viz[currentLayer] || viz.composite_heatmap || viz.original_image;
    activeRadiograph.src = imgSrc;
  }

  // Inference Spinner & Scan Animation
  function showInferenceLoading(show) {
    inferenceOverlay.style.display = show ? "flex" : "none";
    scanLine.style.display = show ? "block" : "none";
  }

  function generateRandomPatientId() {
    const num = Math.floor(1000 + Math.random() * 9000);
    lblPatientId.textContent = `OPG-2026-X${num}`;
  }

  // Training Panel — Web retraining is DISABLED per security policy.
  // Training must be done from the CLI: python -m backend.train
  function setupTraining() {
    if (btnStartTraining) {
      btnStartTraining.disabled = true;
      btnStartTraining.innerHTML = `<i class="fa-solid fa-lock fa-beat-fade"></i> Web Training Disabled`;
      btnStartTraining.title = "Training from the web UI is disabled. Use CLI: python -m backend.train";
    }

    // Show CLI instructions block if present
    const cliInstructBox = document.getElementById("cliInstructBox");
    if (cliInstructBox) cliInstructBox.style.display = "block";

    // Hide the training progress section — not applicable
    if (trainingProgressBox) trainingProgressBox.style.display = "none";
  }

  // Training polling removed — web retraining disabled per security policy.

});
