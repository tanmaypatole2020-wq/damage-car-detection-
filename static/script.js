/**
 * Vehicle Damage Detection AI - Vanilla Frontend Controller
 * Completely offline, zero CDN dependencies.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const dropZonePrompt = document.getElementById("dropZonePrompt");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const fileNameEl = document.getElementById("fileName");
  const fileSizeEl = document.getElementById("fileSize");
  const removeFileBtn = document.getElementById("removeFileBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");

  const uploadSection = document.getElementById("uploadSection");
  const loadingSection = document.getElementById("loadingSection");
  const resultsSection = document.getElementById("resultsSection");

  const errorBox = document.getElementById("errorBox");
  const errorMessage = document.getElementById("errorMessage");
  const errorClose = document.getElementById("errorClose");

  const resultBadge = document.getElementById("resultBadge");
  const confidencePill = document.getElementById("confidencePill");
  const lowConfBanner = document.getElementById("lowConfBanner");
  const suggestionText = document.getElementById("suggestionText");
  const resultOriginalImg = document.getElementById("resultOriginalImg");
  const resultGradcamImg = document.getElementById("resultGradcamImg");
  const probList = document.getElementById("probList");
  const resetBtn = document.getElementById("resetBtn");
  const bottomResetBtn = document.getElementById("bottomResetBtn");

  // Configuration & State
  const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
  const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];
  const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

  let selectedFile = null;

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  function showError(msg) {
    errorMessage.textContent = msg;
    errorBox.classList.remove("hidden");
  }

  function hideError() {
    errorBox.classList.add("hidden");
    errorMessage.textContent = "";
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function validateFile(file) {
    if (!file) {
      return "No file selected.";
    }

    // Check extension
    const ext = file.name.split(".").pop().toLowerCase();
    const typeValid = ALLOWED_EXTENSIONS.includes(ext) || ALLOWED_MIME_TYPES.includes(file.type);
    if (!typeValid) {
      return "Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image.";
    }

    // Check size limit
    if (file.size > MAX_FILE_SIZE_BYTES) {
      return `File size (${formatBytes(file.size)}) exceeds the 10 MB limit. Please choose a smaller photo.`;
    }

    if (file.size === 0) {
      return "Selected file is empty. Please choose a valid car image.";
    }

    return null;
  }

  function handleFileSelected(file) {
    hideError();
    const validationError = validateFile(file);
    if (validationError) {
      showError(validationError);
      clearFileSelection();
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      fileNameEl.textContent = file.name;
      fileSizeEl.textContent = formatBytes(file.size);

      dropZonePrompt.classList.add("hidden");
      previewContainer.classList.remove("hidden");
      analyzeBtn.disabled = false;
    };
    reader.onerror = () => {
      showError("Could not read image file. Please try another image.");
      clearFileSelection();
    };
    reader.readAsDataURL(file);
  }

  function clearFileSelection() {
    selectedFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    fileNameEl.textContent = "";
    fileSizeEl.textContent = "";

    previewContainer.classList.add("hidden");
    dropZonePrompt.classList.remove("hidden");
    analyzeBtn.disabled = true;
  }

  function resetToUpload() {
    clearFileSelection();
    hideError();
    resultsSection.classList.add("hidden");
    loadingSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
  }

  // -------------------------------------------------------------------------
  // Event Listeners: Drag & Drop & Upload
  // -------------------------------------------------------------------------

  errorClose.addEventListener("click", hideError);

  dropZone.addEventListener("click", (e) => {
    // Only trigger file picker if not clicking the remove button
    if (e.target !== removeFileBtn) {
      fileInput.click();
    }
  });

  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFileSelection();
    hideError();
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileSelected(dt.files[0]);
    }
  });

  // -------------------------------------------------------------------------
  // Analyze & Predict
  // -------------------------------------------------------------------------

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      showError("Please select a car photo first.");
      return;
    }

    hideError();

    // Switch to loading state
    uploadSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || `Server responded with status ${response.status}`);
      }

      displayResults(data);
    } catch (err) {
      console.error("Prediction error:", err);
      loadingSection.classList.add("hidden");
      uploadSection.classList.remove("hidden");
      showError(err.message || "An error occurred while analyzing the image. Please try again.");
    }
  });

  // -------------------------------------------------------------------------
  // Render Results
  // -------------------------------------------------------------------------

  function displayResults(data) {
    loadingSection.classList.add("hidden");

    // 1. Predicted Class Badge
    resultBadge.textContent = data.predicted_class;
    resultBadge.className = "badge"; // Reset classes

    const key = data.class_key;
    if (key === "no_damage") {
      resultBadge.classList.add("badge-no-damage");
    } else if (key === "minor_damage") {
      resultBadge.classList.add("badge-minor");
    } else if (key === "severe_damage") {
      resultBadge.classList.add("badge-severe");
    }

    // 2. Confidence Pill
    confidencePill.textContent = `${data.confidence_percentage} Confidence`;

    // 3. Low Confidence Notice
    if (data.is_low_confidence) {
      lowConfBanner.classList.remove("hidden");
    } else {
      lowConfBanner.classList.add("hidden");
    }

    // 4. One-line Suggestion
    suggestionText.textContent = data.suggestion || "Vehicle damage analysis complete.";

    // 5. Images (Original & Grad-CAM)
    resultOriginalImg.src = data.original_image;
    resultGradcamImg.src = data.gradcam_image;

    // 6. Probability Bars for all 3 classes
    probList.innerHTML = "";
    if (Array.isArray(data.probabilities)) {
      data.probabilities.forEach((item) => {
        const itemEl = document.createElement("div");
        itemEl.className = "prob-item";

        const pct = (item.value * 100).toFixed(1);

        itemEl.innerHTML = `
          <div class="prob-labels">
            <span class="prob-name">${item.label}</span>
            <span class="prob-val">${pct}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: 0%; background-color: ${item.color};"></div>
          </div>
        `;

        probList.appendChild(itemEl);

        // Animate progress bar fill smoothly
        requestAnimationFrame(() => {
          setTimeout(() => {
            const fill = itemEl.querySelector(".prob-bar-fill");
            if (fill) fill.style.width = `${pct}%`;
          }, 60);
        });
      });
    }

    // Show Results View
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Reset Buttons
  resetBtn.addEventListener("click", resetToUpload);
  bottomResetBtn.addEventListener("click", resetToUpload);
});
