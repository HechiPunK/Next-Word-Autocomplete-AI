const prompt = document.querySelector("#prompt");
const suggestions = document.querySelector("#suggestions");
const wordCount = document.querySelector("#word-count");
const modelStatus = document.querySelector("#model-status");
const copyButton = document.querySelector("#copy-text");
const deleteButton = document.querySelector("#delete-text");

let requestId = 0;

function updateCount() {
  const words = prompt.value.trim() ? prompt.value.trim().split(/\s+/).length : 0;
  wordCount.textContent = `${words} ${words === 1 ? "WORD" : "WORDS"}`;
}

function renderSuggestions(words, loading = false) {
  suggestions.replaceChildren();
  if (loading) {
    const item = document.createElement("span");
    item.className = "empty-state";
    item.textContent = "Thinking...";
    suggestions.append(item);
    return;
  }
  if (!words.length) {
    const item = document.createElement("span");
    item.className = "empty-state";
    item.textContent = "Suggestions will appear here.";
    suggestions.append(item);
    return;
  }
  words.forEach((word) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion";
    button.textContent = word;
    button.addEventListener("click", () => {
      prompt.value = `${prompt.value.trimEnd()} ${word}`.trimStart();
      updateCount();
      prompt.focus();
      predict();
    });
    suggestions.append(button);
  });
}

async function predict() {
  if (!prompt.value.trim()) {
    renderSuggestions([]);
    return;
  }
  const currentRequest = ++requestId;
  renderSuggestions([], true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: prompt.value, limit: 3 }),
    });
    if (!response.ok) throw new Error("Prediction failed");
    const data = await response.json();
    if (currentRequest === requestId) renderSuggestions(data.suggestions);
  } catch {
    if (currentRequest === requestId) {
      suggestions.textContent = "Model unavailable. Please try again.";
    }
  }
}

prompt.addEventListener("input", updateCount);
prompt.addEventListener("input", () => {
  clearTimeout(window.predictionTimer);
  window.predictionTimer = setTimeout(predict, 180);
});

copyButton.addEventListener("click", async () => {
  if (!prompt.value) return;
  await navigator.clipboard.writeText(prompt.value);
  copyButton.textContent = "COPIED";
  setTimeout(() => { copyButton.textContent = "COPY"; }, 1200);
});

deleteButton.addEventListener("click", () => {
  prompt.value = "";
  updateCount();
  clearTimeout(window.predictionTimer);
  requestId++;
  renderSuggestions([]);
  prompt.focus();
});

fetch("/api/health")
  .then((response) => response.json())
  .then((data) => {
    modelStatus.textContent = "MODEL ONLINE";
    modelStatus.classList.add("online");
    modelStatus.title = data.model;
  })
  .catch(() => {
    modelStatus.textContent = "MODEL OFFLINE";
  });

renderSuggestions([]);
