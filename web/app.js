// Interfaz de la demo del autocompletador.
//
// Contrato con el servidor (src/serve/web_app.py, que NO se modifica):
//   POST /api/predict   cuerpo {"text": "..."}
//   respuesta           {"suggestions": [3 palabras],
//                        "context": [{"word": "...", "known": true|false}, ...],
//                        "latency_ms": numero}
//   Errores HTTP: 400 (sin palabras válidas), 413 (texto muy largo), 415 (no es JSON).

// ---- Evaluación que se muestra en la página: ÚNICO lugar donde editarla ----
// Debe coincidir con logs/final_test_*.json. Cuando exista logs/final_test_pi.json,
// actualizar estos valores con ese archivo y poner device: "a Raspberry Pi 4".
const EVAL = {
  top3: "26.68%",
  predictions: "266,257",
  unknownShare: "8.1%",
  device: "a laptop",
};

const API_URL = "/api/predict";
const DEBOUNCE_MS = 200;
// true = sugerir solo cuando el texto termina en espacio, es decir, cuando la última palabra ya está
// completa (el modelo predice la SIGUIENTE palabra, así que a mitad de palabra no tendría sentido).
const PREDICT_ONLY_AFTER_SPACE = true;

const promptInput = document.querySelector("#prompt");
const suggestions = document.querySelector("#suggestions");
const contextWindow = document.querySelector("#context-window");
const responseTime = document.querySelector("#response-time");
const wordCount = document.querySelector("#word-count");
const modelStatus = document.querySelector("#model-status");
const copyButton = document.querySelector("#copy-text");
const deleteButton = document.querySelector("#delete-text");
const evalNote = document.querySelector("#eval-note");

let requestId = 0;
let debounceTimer;

function setStatus(online) {
  modelStatus.textContent = online ? "MODEL ONLINE" : "MODEL OFFLINE";
  modelStatus.classList.toggle("online", online);
}

function updateCount() {
  const text = promptInput.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  wordCount.textContent = `${words} ${words === 1 ? "WORD" : "WORDS"}`;
}

function contextToken(text, className, title) {
  const element = document.createElement("span");
  element.className = `context-token ${className}`.trim();
  element.textContent = text;
  if (title) element.title = title;
  return element;
}

// tokens = lista del servidor, o null para mostrar la ventana vacía.
// El servidor rellena con "<unk>" cuando hay menos de 3 palabras: eso es relleno, no una palabra desconocida.
function renderContext(tokens) {
  contextWindow.replaceChildren();
  for (const token of tokens || [null, null, null]) {
    if (!token || token.word === "<unk>") {
      contextWindow.append(contextToken("…", "is-empty", "No word here yet"));
    } else if (!token.known) {
      contextWindow.append(contextToken(token.word, "is-unknown", "Outside the model's vocabulary"));
    } else {
      contextWindow.append(contextToken(token.word, "", ""));
    }
  }
}

function renderMessage(text) {
  suggestions.replaceChildren();
  const item = document.createElement("span");
  item.className = "empty-state";
  item.textContent = text;
  suggestions.append(item);
}

function renderSuggestions(words) {
  if (!words || !words.length) {
    renderMessage("No suggestions for this text. Try a different phrase.");
    return;
  }
  suggestions.replaceChildren();
  for (const word of words) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion";
    button.textContent = word;
    button.addEventListener("click", () => {
      promptInput.value = `${promptInput.value.trimEnd()} ${word} `.trimStart();
      updateCount();
      promptInput.focus();
      window.clearTimeout(debounceTimer);
      predict();
    });
    suggestions.append(button);
  }
}

function resetResults() {
  renderMessage("Type a word, then press space to see suggestions.");
  renderContext(null);
  responseTime.textContent = "— ms";
}

function errorMessage(status) {
  if (status === 400) return "Type at least one English word.";
  if (status === 413) return "That text is too long. Shorten it and try again.";
  return "The model did not answer. Try again in a moment.";
}

async function predict() {
  const currentRequest = ++requestId;
  const text = promptInput.value;

  if (!text.trim()) {
    resetResults();
    return;
  }
  if (PREDICT_ONLY_AFTER_SPACE && !/\s$/.test(text)) {
    renderMessage("Press space after a word to see suggestions.");
    renderContext(null);
    responseTime.textContent = "— ms";
    return;
  }

  renderMessage("Thinking...");
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (currentRequest !== requestId) return;   // llegó una respuesta vieja: se descarta
    setStatus(response.status < 500);

    if (!response.ok) {
      renderMessage(errorMessage(response.status));
      renderContext(null);
      responseTime.textContent = "— ms";
      return;
    }

    const data = await response.json();
    if (currentRequest !== requestId) return;
    renderContext(data.context);
    renderSuggestions(data.suggestions);
    responseTime.textContent = Number.isFinite(data.latency_ms) ? `${data.latency_ms.toFixed(1)} ms` : "— ms";
  } catch {
    if (currentRequest !== requestId) return;
    setStatus(false);
    renderMessage("Cannot reach the model. Check that the server is running and try again.");
    renderContext(null);
    responseTime.textContent = "— ms";
  }
}

// Al cargar la página, una consulta de prueba indica si el servidor y el modelo responden.
async function checkModel() {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "i do not" }),
    });
    setStatus(response.ok);
  } catch {
    setStatus(false);
  }
}

async function copyText() {
  if (!promptInput.value) return;
  let copied = false;
  try {
    await navigator.clipboard.writeText(promptInput.value);
    copied = true;
  } catch {
    // Fuera de localhost/HTTPS el navegador no ofrece clipboard: se usa el método antiguo.
    promptInput.select();
    copied = document.execCommand("copy");
    const end = promptInput.value.length;
    promptInput.setSelectionRange(end, end);
  }
  copyButton.textContent = copied ? "COPIED" : "COPY FAILED";
  window.setTimeout(() => { copyButton.textContent = "COPY"; }, 1200);
}

promptInput.addEventListener("input", () => {
  updateCount();
  window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(predict, DEBOUNCE_MS);
});

copyButton.addEventListener("click", copyText);

deleteButton.addEventListener("click", () => {
  promptInput.value = "";
  window.clearTimeout(debounceTimer);
  requestId++;                                  // descarta cualquier respuesta en camino
  updateCount();
  resetResults();
  promptInput.focus();
});

document.querySelectorAll("[data-example]").forEach((chip) => {
  chip.addEventListener("click", () => {
    promptInput.value = `${chip.dataset.example} `;
    updateCount();
    promptInput.focus();
    window.clearTimeout(debounceTimer);
    predict();
  });
});

evalNote.textContent =
  `Held-out test on books the model never saw (Mark Twain, Jack London): ${EVAL.top3} top-3 accuracy ` +
  `(the right word is among the three suggestions) over ${EVAL.predictions} predictions, measured on ${EVAL.device}. ` +
  `Predictions whose correct word is outside the vocabulary (${EVAL.unknownShare}) are not counted.`;

updateCount();
resetResults();
checkModel();
