import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.data.clean import normalize
from src.serve.predictor import Predictor


ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
WEIGHTS_PATH = ROOT / "models" / "mlp_weights_c3_d30_more.npz"
VOCAB_PATH = ROOT / "models" / "vocab.json"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_BYTES = 16_384


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, predictor):
        self.predictor = predictor
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def do_GET(self):
        static_file = STATIC_FILES.get(self.path)
        if static_file is None:
            self._send_json(404, {"error": "No se encontró esa ruta."})
            return

        filename, content_type = static_file
        body = (WEB_DIR / filename).read_bytes()
        self._send_response(200, content_type, body)

    def do_POST(self):
        if self.path != "/api/predict":
            self._send_json(404, {"error": "No se encontró esa ruta."})
            return

        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._send_json(415, {"error": "Envía el texto como JSON."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(400, {"error": "La solicitud no tiene una longitud válida."})
            return
        if content_length < 0:
            self._send_json(400, {"error": "La longitud de la solicitud no es válida."})
            return
        if content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "El texto es demasiado largo."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "El cuerpo debe ser un JSON válido."})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            self._send_json(400, {"error": "Incluye un campo de texto válido."})
            return

        text = payload["text"]
        words = normalize(text).split()
        if not words:
            self._send_json(400, {"error": "Escribe al menos una palabra en inglés."})
            return

        predictor = self.server.predictor
        context_words = words[-predictor.context:]
        context_words = ["<unk>"] * (predictor.context - len(context_words)) + context_words
        context = [
            {"word": word, "known": word != "<unk>" and word in predictor.word2id}
            for word in context_words
        ]

        started = time.perf_counter()
        predictions = predictor.predict_top3(text)
        latency_ms = (time.perf_counter() - started) * 1000
        self._send_json(200, {
            "suggestions": predictions,
            "context": context,
            "latency_ms": latency_ms,
        })

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_response(status, "application/json; charset=utf-8", body)

    def _send_response(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        print(f"[web-demo] {self.address_string()} - {format_string % args}")


def main():
    parser = argparse.ArgumentParser(description="Inicia la demo web local del autocompletador.")
    parser.add_argument("--port", type=int, default=8000, help="Puerto local (por defecto: 8000)")
    args = parser.parse_args()

    predictor = Predictor(WEIGHTS_PATH, VOCAB_PATH)
    server = DemoServer(("127.0.0.1", args.port), predictor)
    print(f"Demo disponible en http://127.0.0.1:{args.port} (Ctrl+C para detener)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
