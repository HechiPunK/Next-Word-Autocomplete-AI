import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.ngram_baseline import NgramBaseline
from src.eval.metrics import top3_accuracy

PROC = ROOT / "data" / "processed"
vocab = json.loads((PROC / "vocab.json").read_text(encoding="utf-8"))

X_train, y_train = np.load(PROC / "X_train.npy"), np.load(PROC / "y_train.npy")
X_val, y_val = np.load(PROC / "X_val.npy"), np.load(PROC / "y_val.npy")

model = NgramBaseline(len(vocab)).fit(X_train, y_train)

results = []
for order in (1, 2, 3):
    for min_count in ((1,) if order == 1 else (1, 2, 3, 5, 10)):
        preds = model.predict_top3(X_val, order, min_count)
        acc, n, excl = top3_accuracy(preds, y_val)
        results.append({"orden": order, "min_count": min_count,
                        "top3_val": round(acc, 4), "evaluados": n, "excluidos_unk": excl})
        print(f"orden {order} | min_count {min_count:>2} | top-3 en val = {acc:.1%}")

best = max(results, key=lambda r: r["top3_val"])
print(f"\nMejor configuración: orden {best['orden']}, min_count {best['min_count']} "
      f"-> {best['top3_val']:.1%}")

preds = model.predict_top3(X_val, best["orden"], best["min_count"])
print("\nEjemplos (val):")
for i in (1000, 5000, 20000, 50000):
    ctx = " ".join(vocab[j] for j in X_val[i])
    print(f"  {ctx!r} -> {[vocab[j] for j in preds[i]]}   (real: {vocab[y_val[i]]})")

(ROOT / "logs").mkdir(exist_ok=True)
(ROOT / "logs" / "baseline_val.json").write_text(
    json.dumps({"mejor": best, "todas": results}, indent=2), encoding="utf-8")