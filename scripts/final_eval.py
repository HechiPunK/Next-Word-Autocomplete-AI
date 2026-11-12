import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import encode, make_windows
from src.eval.metrics import top3_accuracy
from src.serve.predictor import Predictor

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="c3_d30_more")        # qué modelo evaluar
parser.add_argument("--nombre", default="laptop")          # etiqueta del dispositivo: laptop o pi
parser.add_argument("--baseline", action="store_true")     # también evalúa el baseline (solo laptop)
args = parser.parse_args()

PROC = ROOT / "data" / "processed"
predictor = Predictor(ROOT / "models" / f"mlp_weights_{args.tag}.npz", ROOT / "models" / "vocab.json")


def device_name():
    try:
        return Path("/proc/device-tree/model").read_text().strip("\x00 \n")   # modelo de la Raspberry
    except Exception:
        return platform.platform()


def net_top3(X):
    out = []
    for i in range(0, len(X), 1024):
        s = predictor.logits(X[i:i + 1024]).copy()
        s[:, 0] = -np.inf                                    # nunca sugerir <unk>
        top = np.argpartition(-s, 3, axis=1)[:, :3]
        order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
        out.append(np.take_along_axis(top, order, axis=1))
    return np.concatenate(out)


# Ejemplos de test: una ventana de contexto por cada posición de cada libro
books = {}
for f in sorted((PROC / "test").glob("*.txt")):
    ids = encode(f.read_text(encoding="utf-8").split(), predictor.word2id)
    books[f.name] = make_windows(ids, predictor.context)


def evaluate(predict_fn):
    per_book, P, Y = {}, [], []
    for name, (X, y) in books.items():
        p = predict_fn(X)
        acc, n, excl = top3_accuracy(p, y)
        per_book[name] = {"top3": round(acc, 4), "evaluados": n, "excluidos_unk": excl}
        P.append(p)
        Y.append(y)
    acc, n, excl = top3_accuracy(np.concatenate(P), np.concatenate(Y))
    margin = 1.96 * (acc * (1 - acc) / n) ** 0.5             # aproximado
    return {"top3": round(acc, 4), "margen_95": round(margin, 4), "evaluados": n,
            "excluidos_unk": excl, "por_libro": per_book}


# Velocidad de una sugerencia real (texto -> 3 palabras)
phrases = ["i do not", "it was a dark and", "she looked at him with"]
t0 = time.perf_counter()
for i in range(300):
    predictor.predict_top3(phrases[i % 3])
latency_ms = (time.perf_counter() - t0) / 300 * 1000

result = {
    "fecha": datetime.now().isoformat(timespec="seconds"),
    "dispositivo": device_name(),
    "modelo": args.tag,
    "red": evaluate(net_top3),
    "latencia_ms_por_sugerencia": round(latency_ms, 2),
    "ejemplos": {p: predictor.predict_top3(p) for p in phrases},
}

if args.baseline:
    from src.model.ngram_baseline import NgramBaseline
    cfg = json.loads((ROOT / "logs" / "baseline_val.json").read_text(encoding="utf-8"))["mejor"]
    nb = NgramBaseline(len(predictor.vocab)).fit(np.load(PROC / "X_train.npy"), np.load(PROC / "y_train.npy"))
    result["baseline"] = evaluate(lambda X: nb.predict_top3(X, cfg["orden"], cfg["min_count"]))
    result["baseline"]["config"] = cfg

(ROOT / "logs").mkdir(exist_ok=True)
out = ROOT / "logs" / f"final_test_{args.nombre}.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

r = result["red"]
print(f"Dispositivo: {result['dispositivo']}")
print(f"RED:      top-3 en test = {r['top3']:.1%} ± {r['margen_95']:.1%}  "
      f"({r['evaluados']:,} ejemplos evaluados, {r['excluidos_unk']:,} excluidos por <unk>)")
if args.baseline:
    b = result["baseline"]
    print(f"BASELINE: top-3 en test = {b['top3']:.1%} ± {b['margen_95']:.1%}")
print(f"Velocidad: {latency_ms:.1f} ms por sugerencia")
for name, v in r["por_libro"].items():
    print(f"  {name:<12} top-3 = {v['top3']:.1%}  ({v['evaluados']:,} ejemplos)")
print(f"\nGuardado en {out}")