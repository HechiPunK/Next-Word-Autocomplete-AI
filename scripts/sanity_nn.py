import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.mlp import MLPLanguageModel

PROC = ROOT / "data" / "processed"
vocab = json.loads((PROC / "vocab.json").read_text(encoding="utf-8"))
X_train, y_train = np.load(PROC / "X_train.npy"), np.load(PROC / "y_train.npy")

# 1) Chequeo de gradientes: derivada numérica vs. la de nuestro backprop
print("== 1) Chequeo de gradientes (deben coincidir) ==")
rng = np.random.default_rng(0)
small = MLPLanguageModel(30, context=3, emb_dim=4, hidden=8, dtype=np.float64)
Xs, ys = rng.integers(0, 30, (6, 3)), rng.integers(0, 30, 6)
_, grads = small.loss_and_grads(Xs, ys)
eps = 1e-5
for name, P in small.params.items():
    for _ in range(3):
        idx = tuple(rng.integers(0, s) for s in P.shape)
        old = P[idx]
        P[idx] = old + eps
        l1, _ = small.loss_and_grads(Xs, ys)
        P[idx] = old - eps
        l2, _ = small.loss_and_grads(Xs, ys)
        P[idx] = old
        print(f"  {name:2} numérico {(l1 - l2) / (2 * eps):+.6e} | analítico {grads[name][idx]:+.6e}")

# 2) Sobreajuste a propósito: si no puede memorizar 512 ejemplos, hay un bug
print("\n== 2) Memorizar 512 ejemplos (la pérdida debe bajar de ~9 a casi 0) ==")
model = MLPLanguageModel(len(vocab), hidden=256)
Xb, yb = X_train[:512], y_train[:512]
for step in range(301):
    loss, g = model.loss_and_grads(Xb, yb)
    model.adam_step(g, lr=3e-3)
    if step % 50 == 0:
        print(f"  paso {step:3d} | pérdida {loss:.4f}")

# 3) Tiempo del modelo de tamaño real (2,048 ocultas + 8,000 salida)
print("\n== 3) Tiempo por paso, modelo real ==")
full = MLPLanguageModel(len(vocab), hidden=2048)
B = 256
Xb, yb = X_train[:B], y_train[:B]
t0 = time.time()
for _ in range(5):
    loss, g = full.loss_and_grads(Xb, yb)
    full.adam_step(g)
dt = (time.time() - t0) / 5
print(f"  {dt:.2f} s por paso -> ~{dt * (len(X_train) // B) / 60:.0f} min por época")