import json
from pathlib import Path

import numpy as np

from src.data.clean import normalize


class Predictor:
    """Inferencia solo con NumPy (es lo único que necesita la Raspberry)."""

    def __init__(self, weights_path, vocab_path):
        w = np.load(weights_path)
        self.E, self.W1, self.b1 = w["E"], w["W1"], w["b1"]
        self.W2, self.b2 = w["W2"], w["b2"]
        self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.word2id = {word: i for i, word in enumerate(self.vocab)}
        self.context = self.W1.shape[0] // self.E.shape[1]   # entradas / tamaño del embedding

    def logits(self, ids):
        """ids: arreglo (lote, contexto) de enteros. Devuelve un puntaje por palabra."""
        x = self.E[ids].reshape(len(ids), -1)
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def predict_top3(self, text):
        words = normalize(text).split()
        ids = [self.word2id.get(word, 0) for word in words][-self.context:]
        ids = [0] * (self.context - len(ids)) + ids
        scores = self.logits(np.array([ids]))[0].copy()
        scores[0] = -np.inf                          # nunca sugerir <unk>
        best = np.argsort(-scores)[:3]
        return [self.vocab[i] for i in best]