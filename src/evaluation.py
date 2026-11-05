"""
Valutazione: metriche per run, aggregazione e confronto statistico tra modelli.

Metriche (macro-medie sulle classi, quindi robuste allo sbilanciamento):
  Precision, Recall, F1  e  G-Mean = media geometrica delle recall per classe
  (vale 0 se anche una sola classe non viene mai riconosciuta).

Confronto statistico (Demsar, 2006): test di Friedman sulle F1 dei modelli nei run
(blocchi = run) e, se significativo, post-hoc di Nemenyi con differenza critica.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             recall_score)

METRICS = ["accuracy", "precision", "recall", "f1", "gmean"]
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1 macro", "gmean": "G-Mean"}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    labels = list(range(n_classes))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  average="macro", zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "gmean": float(np.prod(recalls) ** (1.0 / n_classes)),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def summarize(runs: pd.DataFrame, model_order: Sequence[str]) -> pd.DataFrame:
    """Media e deviazione standard (ddof=1) di ogni metrica per modello."""
    g = runs.groupby("model")[METRICS]
    mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    out = pd.concat({"mean": mean, "std": std}, axis=1)
    return out.loc[[m for m in model_order if m in out.index]]


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """Friedman + Nemenyi.  scores: righe = run (blocchi), colonne = modelli."""
    n, k = scores.shape
    res: Dict = {"n_runs": n, "n_models": k, "alpha": alpha}
    if n < 3 or k < 3:
        res.update(statistic=np.nan, pvalue=np.nan, mean_ranks=None, nemenyi=None, cd=np.nan)
        return res

    stat, p = stats.friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
    ranks = scores.rank(axis=1, ascending=False, method="average")   # rango 1 = migliore
    mean_ranks = ranks.mean(axis=0)

    se = np.sqrt(k * (k + 1) / (6.0 * n))
    diff = np.abs(mean_ranks.to_numpy()[:, None] - mean_ranks.to_numpy()[None, :])
    q = diff / se
    # p-value di Nemenyi dalla distribuzione del range studentizzato (df = infinito)
    pvals = stats.studentized_range.sf(q * np.sqrt(2.0), k, np.inf)
    np.fill_diagonal(pvals, 1.0)
    cd = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0) * se)

    res.update(statistic=float(stat), pvalue=float(p), mean_ranks=mean_ranks.sort_values(),
               nemenyi=pd.DataFrame(pvals, index=scores.columns, columns=scores.columns), cd=cd)
    return res
