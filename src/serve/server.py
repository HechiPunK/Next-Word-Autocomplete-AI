import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = Path(__file__).with_name("static")
VOCAB_PATH = ROOT / "data" / "processed" / "vocab.json"
WEIGHTS_PATH = ROOT / "models" / "mlp_weights_c3_d50.npz"


class Predictor:
    def __init__(self):
        vocab_data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        self.id_to_word = vocab_data
        self.word_to_id = {word: index for index, word in enumerate(vocab_data)}
        weights = np.load(WEIGHTS_PATH)
        self.embeddings = weights["E"]
        self.w1 = weights["W1"]
        self.b1 = weights["b1"]
        self.w2 = weights["W2"]
        self.b2 = weights["b2"]
        self.context = self.w1.shape[0] // self.embeddings.shape[1]

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z' ]", " ", text)
        return re.sub(r"\s+", " ", text).strip().split()

    def predict(self, text, limit=3):
        words = self.tokenize(text)[-self.context:]
        context_ids = [
            self.word_to_id.get(word, self.word_to_id["<unk>"])
            for word in words
        ]
        context_ids = ([self.word_to_id["<unk>"]] * (self.context - len(context_ids))) + context_ids

        embedded = self.embeddings[np.array(context_ids)].reshape(1, -1)
        hidden = np.maximum(embedded @ self.w1 + self.b1, 0)
        logits = hidden @ self.w2 + self.b2
        logits[0, self.word_to_id["<unk>"]] = -np.inf

        count = min(limit, len(logits[0]))
        ranked = np.argpartition(-logits[0], count - 1)[:count]
        ranked = ranked[np.argsort(-logits[0, ranked])]
        return [self.id_to_word[index] for index in ranked]


predictor = Predictor()


class AppHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(200, {
                "status": "online",
                "model": WEIGHTS_PATH.name,
                "context": predictor.context,
            })
            return

        requested = PUBLIC / ("index.html" if path in ("/", "/index.html") else path.lstrip("/"))
        try:
            body = requested.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            self.send_response(404)
            self.end_headers()
            return

        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }
        self.send_response(200)
        self.send_header("Content-Type", content_types.get(requested.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            return

        requested = PUBLIC / ("index.html" if path in ("/", "/index.html") else path.lstrip("/"))
        if not requested.is_file():
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            length = min(int(self.headers.get("Content-Length", 0)), 16_384)
            payload = json.loads(self.rfile.read(length))
            text = str(payload.get("text", ""))
            limit = max(1, min(int(payload.get("limit", 3)), 5))
            suggestions = predictor.predict(text, limit)
            self.send_json(200, {"suggestions": suggestions})
        except (json.JSONDecodeError, TypeError, ValueError):
            self.send_json(400, {"error": "Send valid JSON with a text field."})

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8000), AppHandler)
    print(f"Next Word AI running at http://localhost:8000 (model context: {predictor.context})")
    server.serve_forever()
