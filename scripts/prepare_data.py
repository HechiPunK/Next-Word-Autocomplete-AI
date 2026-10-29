import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clean import clean_book

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# Busca los libros en data/raw sin importar en qué subcarpeta estén
index = {}
for p in RAW.rglob("pg*.txt"):
    if p.name in index:
        print(f"AVISO: {p.name} está repetido en {index[p.name].parent.name} y {p.parent.name}")
    index[p.name] = p
print(f"Buscando en {RAW} -> {len(index)} archivos encontrados\n")

totals = {"train": 0, "val": 0, "test": 0}

with open(ROOT / "data" / "manifest.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        split = row["split"].strip()
        if split not in totals:
            print(f"SPLIT INVÁLIDO '{split}': {row['title']}")
            continue

        src = index.get(f"pg{row['id']}.txt")
        if src is None:
            print(f"FALTA: pg{row['id']}.txt ({row['title']})")
            continue

        # Avisa si la carpeta donde lo pusiste no coincide con el manifest
        if src.parent.name != split:
            print(f"AVISO: {src.name} está en raw/{src.parent.name} pero el manifest dice {split}")

        text = clean_book(src.read_text(encoding="utf-8-sig", errors="ignore"))
        (OUT / split).mkdir(parents=True, exist_ok=True)
        (OUT / split / src.name).write_text(text, encoding="utf-8")

        n = len(text.split())
        totals[split] += n
        print(f"  {split:5} {row['author']:<22} {row['title'][:35]:<35} {n:>9,}")

print()
for split, n in totals.items():
    print(f"== {split}: {n:,} palabras")