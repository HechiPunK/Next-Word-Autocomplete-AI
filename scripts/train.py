import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--dropout", type=float, default=0.2)   # prob. de apagar neuronas al entrenar
parser.add_argument("--wd", type=float, default=0.0)        # weight decay (penaliza pesos grandes)
parser.add_argument("--lr", type=float, default=1e-3)       # tamaño del paso de corrección
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--tag", default="base")                # nombre del experimento
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
(ROOT / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)

EMB_DIM, HIDDEN, BATCH, SEED = 64, 2048, 1024, 42

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

vocab = json.loads((PROC / "vocab.json").read_text(encoding="utf-8"))
V = len(vocab)


def load(split):
    X = torch.from_numpy(np.load(PROC / f"X_{split}.npy").astype(np.int64)).to(device)
    y = torch.from_numpy(np.load(PROC / f"y_{split}.npy").astype(np.int64)).to(device)
    return X, y


X_train, y_train = load("train")
X_val, y_val = load("val")
CONTEXT = X_train.shape[1]
print(f"Experimento '{args.tag}': contexto={CONTEXT}, dropout={args.dropout}, wd={args.wd}, lr={args.lr}")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, EMB_DIM)
        self.fc1 = nn.Linear(CONTEXT * EMB_DIM, HIDDEN)
        self.drop = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(HIDDEN, V)

    def forward(self, X):
        x = self.emb(X).flatten(1)
        h = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(h)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    loss_sum, hits, total = 0.0, 0, 0
    for i in range(0, len(X), 4096):
        xb, yb = X[i:i + 4096], y[i:i + 4096]
        logits = model(xb)
        loss_sum += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        logits[:, 0] = float("-inf")
        top3 = logits.topk(3, dim=1).indices
        keep = yb != 0
        hits += (top3[keep] == yb[keep, None]).any(dim=1).sum().item()
        total += keep.sum().item()
    model.train()
    return loss_sum / len(X), hits / total


def export_numpy(model, path):
    g = lambda t: np.ascontiguousarray(t.detach().cpu().numpy())
    np.savez(path, E=g(model.emb.weight), W1=g(model.fc1.weight.T), b1=g(model.fc1.bias),
             W2=g(model.fc2.weight.T), b2=g(model.fc2.bias))


model = MLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
loss_fn = nn.CrossEntropyLoss()

log_rows, best, best_epoch, min_val_loss = [], 0.0, 0, float("inf")
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    perm = torch.randperm(len(X_train), device=device)
    running = torch.zeros((), device=device)
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.detach() * len(idx)
    train_loss = running.item() / len(X_train)
    val_loss, val_top3 = evaluate(model, X_val, y_val)
    min_val_loss = min(min_val_loss, val_loss)
    mins = (time.time() - t0) / 60
    print(f"Época {epoch} | train {train_loss:.3f} | val {val_loss:.3f} | "
          f"top-3 val {val_top3:.1%} | {mins:.1f} min")
    log_rows.append([epoch, round(train_loss, 4), round(val_loss, 4), round(val_top3, 4), round(mins, 2)])

    if val_top3 > best:
        best, best_epoch = val_top3, epoch
        torch.save(model.state_dict(), ROOT / "models" / "checkpoints" / f"best_{args.tag}.pt")
        export_numpy(model, ROOT / "models" / f"mlp_weights_{args.tag}.npz")
        print("   -> guardado (mejor hasta ahora)")

with open(ROOT / "logs" / f"train_log_{args.tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoca", "train_loss", "val_loss", "top3_val", "minutos"])
    w.writerows(log_rows)

exp_file = ROOT / "logs" / "experiments.csv"
is_new = not exp_file.exists()
with open(exp_file, "a", newline="") as f:
    w = csv.writer(f)
    if is_new:
        w.writerow(["tag", "contexto", "dropout", "weight_decay", "lr",
                    "mejor_top3_val", "mejor_epoca", "val_loss_minima"])
    w.writerow([args.tag, CONTEXT, args.dropout, args.wd, args.lr,
                round(best, 4), best_epoch, round(min_val_loss, 4)])

print(f"\nMejor top-3 en val: {best:.1%} en la época {best_epoch}  (baseline de n-gramas: 27.6%)")