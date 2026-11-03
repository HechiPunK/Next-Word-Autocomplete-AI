import numpy as np


class NgramBaseline:
    """Predictor por conteo con retroceso: 3 palabras -> 2 -> 1 -> más frecuentes."""

    def __init__(self, vocab_size, max_order=3, unk_id=0):
        self.V = vocab_size
        self.max_order = max_order
        self.unk = unk_id
        self.tables = {}
        self.unigram = None

    def _keys(self, X, k):
        """Las últimas k palabras de cada ventana, codificadas como un solo entero."""
        key = np.zeros(len(X), dtype=np.int64)
        for j in range(X.shape[1] - k, X.shape[1]):
            key = key * self.V + X[:, j]
        return key

    def fit(self, X, y):
        keep = y != self.unk                      # nunca se sugiere <unk>
        X, y = X[keep], y[keep]
        self.unigram = np.argsort(-np.bincount(y, minlength=self.V))[:3].astype(np.int32)

        for k in range(1, self.max_order + 1):
            pair = self._keys(X, k) * self.V + y
            uniq, cnt = np.unique(pair, return_counts=True)
            ctx, nxt = uniq // self.V, uniq % self.V
            order = np.lexsort((-cnt, ctx))       # por contexto, mayor conteo primero
            ctx, nxt, cnt = ctx[order], nxt[order], cnt[order]
            keys, starts, sizes = np.unique(ctx, return_index=True, return_counts=True)
            totals = np.add.reduceat(cnt, starts)  # cuántas veces apareció cada contexto
            top3 = np.full((len(keys), 3), -1, dtype=np.int32)
            for r in range(3):
                ok = sizes > r
                top3[ok, r] = nxt[starts[ok] + r]
            self.tables[k] = (keys, top3, totals)
        return self

    def predict_top3(self, X, order=3, min_count=1):
        preds = np.tile(self.unigram, (len(X), 1))
        for k in range(1, order + 1):
            keys, top3, totals = self.tables[k]
            need = min_count if k > 1 else 1
            q = self._keys(X, k)
            idx = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
            found = (keys[idx] == q) & (totals[idx] >= need)
            row = top3[idx]
            merged = np.where(row == -1, preds, row)   # rellena huecos con el orden menor
            preds = np.where(found[:, None], merged, preds)
        return preds