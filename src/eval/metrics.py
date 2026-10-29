import numpy as np


def top3_accuracy(top3_preds, y_true, unk_id=0):
    """
    top3_preds: array (n, 3) con los IDs sugeridos, de mayor a menor probabilidad
    y_true:     array (n,) con el ID correcto
    Devuelve (accuracy, ejemplos evaluados, ejemplos excluidos por ser <unk>)
    """
    mask = y_true != unk_id
    hits = (top3_preds[mask] == y_true[mask][:, None]).any(axis=1)
    return float(hits.mean()), int(mask.sum()), int((~mask).sum())