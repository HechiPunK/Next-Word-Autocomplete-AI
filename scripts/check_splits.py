import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

with open(ROOT / "data" / "manifest.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# 1) IDs repetidos
ids = [r["id"] for r in rows]
dups = sorted({i for i in ids if ids.count(i) > 1})
print("IDs repetidos:", dups or "ninguno")

# 2) Autores en más de un split
authors = defaultdict(set)
for r in rows:
    authors[r["author"].strip()].add(r["split"].strip())
mixed = {a: sorted(s) for a, s in authors.items() if len(s) > 1}
print("Autores en varios splits:", mixed or "ninguno")

# 3) Solapamiento de 8-gramas con train (detecta libros duplicados)
N = 8

def grams(words):
    return {hash(tuple(words[i:i + N])) for i in range(len(words) - N + 1)}

train_grams = set()
for f in (PROC / "train").glob("*.txt"):
    train_grams |= grams(f.read_text(encoding="utf-8").split())

print(f"\nSolapamiento de {N}-gramas con train:")
for split in ("val", "test"):
    for f in sorted((PROC / split).glob("*.txt")):
        g = grams(f.read_text(encoding="utf-8").split())
        print(f"  {split:5} {f.name:<14} {len(g & train_grams) / max(len(g), 1):6.2%}")