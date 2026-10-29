import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import read_split, build_vocab, encode, make_windows, UNK

VOCAB_SIZE = 8000
CONTEXT = 3
PROC = ROOT / "data" / "processed"

books = {s: read_split(PROC / s) for s in ("train", "val", "test")}

vocab = build_vocab(books["train"], VOCAB_SIZE)   # SOLO train
word2id = {w: i for i, w in enumerate(vocab)}
(PROC / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
print(f"Vocabulario: {len(vocab):,} palabras (incluye {UNK})\n")

data = {}
for split, split_books in books.items():
    parts = [make_windows(encode(b, word2id), CONTEXT) for b in split_books]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    np.save(PROC / f"X_{split}.npy", X)
    np.save(PROC / f"y_{split}.npy", y)
    data[split] = (X, y)
    print(f"{split:5} ejemplos: {len(y):>10,} | objetivos <unk>: {(y == 0).mean():.1%}")

X, y = data["train"]
print("\nEjemplo de train:", [vocab[i] for i in X[1000]], "->", vocab[y[1000]])