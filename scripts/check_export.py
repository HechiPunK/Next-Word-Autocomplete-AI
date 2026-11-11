import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serve.predictor import Predictor
from src.eval.metrics import top3_accuracy

TAG = "c3_d30_more"
PROC = ROOT / "data" / "processed"

p = Predictor(ROOT / "models" / f"mlp_weights_{TAG}.npz", PROC / "vocab.json")
X, y = np.load(PROC / "X_val.npy"), np.load(PROC / "y_val.npy")
assert X.shape[1] == p.context, f"Dataset con contexto {X.shape[1]}, modelo con {p.context}"

preds = []
for i in range(0, len(X), 2048):
    s = p.logits(X[i:i + 2048]).copy()
    s[:, 0] = -np.inf
    top = np.argpartition(-s, 3, axis=1)[:, :3]
    order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
    preds.append(np.take_along_axis(top, order, axis=1))

acc, n, excl = top3_accuracy(np.concatenate(preds), y)
print(f"top-3 en val con NumPy: {acc:.1%} ({n:,} evaluados)  <- debe ser ~29.8%")

for text in ["i do not", "it was a dark and", "she looked at him with"]:
    print(f"{text!r} -> {p.predict_top3(text)}")