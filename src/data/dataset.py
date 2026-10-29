from collections import Counter
from pathlib import Path

import numpy as np

UNK = "<unk>"


def read_split(folder):
    """Devuelve una lista de libros; cada libro es una lista de palabras."""
    return [f.read_text(encoding="utf-8").split() for f in sorted(Path(folder).glob("*.txt"))]


def build_vocab(books, size):
    """Las (size-1) palabras más frecuentes de train, más <unk> en el índice 0."""
    counts = Counter(w for book in books for w in book)
    return [UNK] + [w for w, _ in counts.most_common(size - 1)]


def encode(book, word2id):
    unk = word2id[UNK]
    return np.array([word2id.get(w, unk) for w in book], dtype=np.int32)


def make_windows(ids, context):
    """[w1..wN] -> w(N+1), deslizando de uno en uno."""
    n = len(ids) - context
    X = np.stack([ids[i:i + n] for i in range(context)], axis=1)
    y = ids[context:]
    return X, y