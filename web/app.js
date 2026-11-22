const form = document.querySelector("#demo-form");
const promptInput = document.querySelector("#prompt");
const suggestions = document.querySelector("#suggestions");
const contextWindow = document.querySelector("#context-window");
const responseTime = document.querySelector("#response-time");
const errorMessage = document.querySelector("#error-message");
const charCount = document.querySelector("#char-count");
const suggestionArea = document.querySelector(".suggestion-area");

let debounceTimer;
let activeRequest;

function renderContext(tokens) {
  contextWindow.replaceChildren();
  for (const token of tokens) {
    const element = document.createElement("span");
    element.className = `context-token${token.known ? "" : " is-unknown"}`;
    element.textContent = token.word;
    element.title = token.known ? token.word : "Palabra fuera del vocabulario";
    contextWindow.append(element);
  }
}

function renderSuggestions(words) {
  suggestions.replaceChildren();
  words.forEach((word, index) => {
    const button = document.createElement("button");
    button.className = "suggestion-button";
    button.type = "button";
    button.setAttribute("aria-label", `Añadir ${word} a la frase`);

    const label = document.createElement("strong");
    label.textContent = word;
    const rank = document.createElement("span");
    rank.textContent = `0${index + 1}`;
    button.append(label, rank);
    button.addEventListener("click", () => {
      const current = promptInput.value.trimEnd();
      promptInput.value = current ? `${current} ${word}` : word;
      updateCharacterCount();
      promptInput.focus();
      requestSuggestions();
    });
    suggestions.append(button);
  });
}

function updateCharacterCount() {
  charCount.textContent = `${promptInput.value.length} / 500`;
}

function showLoading(message, withSpinner) {
  suggestions.replaceChildren();
  const state = document.createElement("div");
  state.className = "loading-state";
  if (withSpinner) {
    const spinner = document.createElement("span");
    spinner.className = "loader";
    state.append(spinner);
  }
  state.append(document.createTextNode(message));
  suggestions.append(state);
}

async function requestSuggestions() {
  if (activeRequest) activeRequest.abort();
  activeRequest = new AbortController();
  const controller = activeRequest;
  const text = promptInput.value.trim();

  errorMessage.hidden = true;
  if (!text) {
    showLoading("Escribe una frase para ver sugerencias.", false);
    contextWindow.replaceChildren(...Array.from({ length: 3 }, () => {
      const token = document.createElement("span");
      token.className = "context-token is-empty";
      token.textContent = "…";
      return token;
    }));
    responseTime.textContent = "— ms";
    activeRequest = undefined;
    suggestionArea.setAttribute("aria-busy", "false");
    return;
  }

  suggestionArea.setAttribute("aria-busy", "true");
  showLoading("Calculando sugerencias…", true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    const result = await response.json();
    if (activeRequest !== controller) return;
    if (!response.ok) throw new Error(result.error || "No se pudieron obtener sugerencias.");

    renderContext(result.context);
    renderSuggestions(result.suggestions);
    responseTime.textContent = `${result.latency_ms.toFixed(2)} ms`;
  } catch (error) {
    if (error.name === "AbortError") return;
    if (activeRequest !== controller) return;
    suggestions.replaceChildren();
    errorMessage.textContent = error.message || "No se pudo conectar con el modelo.";
    errorMessage.hidden = false;
  } finally {
    if (activeRequest === controller) {
      activeRequest = undefined;
      suggestionArea.setAttribute("aria-busy", "false");
    }
  }
}

function schedulePrediction() {
  updateCharacterCount();
  window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(requestSuggestions, 250);
}

promptInput.addEventListener("input", schedulePrediction);
form.addEventListener("submit", (event) => {
  event.preventDefault();
  window.clearTimeout(debounceTimer);
  requestSuggestions();
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.example;
    updateCharacterCount();
    requestSuggestions();
  });
});

updateCharacterCount();
requestSuggestions();
