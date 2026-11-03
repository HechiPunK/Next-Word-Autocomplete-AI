import numpy as np


class MLPLanguageModel:
    """Embeddings -> capa oculta ReLU -> softmax. Backprop escrito a mano."""

    def __init__(self, vocab_size, context=3, emb_dim=64, hidden=2048, seed=42, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.V, self.C, self.d, self.H = vocab_size, context, emb_dim, hidden
        n_in = context * emb_dim
        self.params = {
            "E":  rng.normal(0, 0.1, (vocab_size, emb_dim)),
            "W1": rng.normal(0, 1, (n_in, hidden)) * np.sqrt(2.0 / n_in),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1, (hidden, vocab_size)) * np.sqrt(1.0 / hidden),
            "b2": np.zeros(vocab_size),
        }
        self.params = {k: v.astype(dtype) for k, v in self.params.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def forward(self, X):
        p = self.params
        x = p["E"][X].reshape(len(X), -1)      # (B, C*d): embeddings concatenados
        z1 = x @ p["W1"] + p["b1"]
        h = np.maximum(z1, 0)                  # ReLU
        logits = h @ p["W2"] + p["b2"]
        return logits, (x, z1, h)

    @staticmethod
    def softmax(logits):
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def loss_and_grads(self, X, y):
        p = self.params
        B = len(X)
        logits, (x, z1, h) = self.forward(X)
        probs = self.softmax(logits)
        loss = float(-np.log(probs[np.arange(B), y] + 1e-9).mean())

        d_logits = probs
        d_logits[np.arange(B), y] -= 1
        d_logits /= B

        grads = {"W2": h.T @ d_logits, "b2": d_logits.sum(axis=0)}
        dz1 = (d_logits @ p["W2"].T) * (z1 > 0)
        grads["W1"] = x.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        dx = dz1 @ p["W1"].T
        dE = np.zeros_like(p["E"])
        np.add.at(dE, X, dx.reshape(B, self.C, self.d))   # suma el gradiente a cada palabra usada
        grads["E"] = dE
        return loss, grads

    def adam_step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict_top3(self, X, unk_id=0):
        logits, _ = self.forward(X)
        logits[:, unk_id] = -np.inf             # nunca sugerir <unk>
        top3 = np.argpartition(-logits, 3, axis=1)[:, :3]
        order = np.argsort(-np.take_along_axis(logits, top3, axis=1), axis=1)
        return np.take_along_axis(top3, order, axis=1)