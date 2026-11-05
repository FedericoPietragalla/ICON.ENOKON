"""
Feature "OWL": appartenenza di ogni vino ai concetti della conoscenza di background.

Le definizioni dei concetti stanno in src/knowledge.py e coincidono (per costruzione
e grazie a tests/test_consistency.py) con quelle scritte in ontology/wine.owl.
L'appartenenza è una funzione deterministica delle misure: nessun parametro viene
stimato dai dati, quindi non c'è data leakage tra train e test.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

import config as cfg
from src.knowledge import (CONCEPTS, FEATURE_CONCEPTS, definition_str, evaluate_concepts)


def build_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """Tutti i concetti (anche quelli ausiliari) come DataFrame booleano."""
    return evaluate_concepts(df)


def owl_feature_matrix(concepts: pd.DataFrame) -> np.ndarray:
    """Matrice binaria (n x |FEATURE_CONCEPTS|) delle sole feature usate dai modelli."""
    return concepts[FEATURE_CONCEPTS].to_numpy(dtype=np.uint8)


def owl_feature_names() -> List[str]:
    return list(FEATURE_CONCEPTS)


def coverage_by_class(concepts: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """% di vini di ciascuna classe che soddisfano ciascun concetto (+ colonna 'tutti')."""
    rows = {}
    for name in FEATURE_CONCEPTS:
        col = concepts[name].to_numpy(dtype=bool)
        row = {cfg.CLASS_NAMES[k]: 100.0 * col[y == k].mean() if np.any(y == k) else np.nan
               for k in range(len(cfg.CLASS_NAMES))}
        row["tutti"] = 100.0 * col.mean()
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def concept_catalog(concepts: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Catalogo dei concetti: famiglia, definizione formale, fondamento (+ copertura)."""
    rows = []
    for c in CONCEPTS:
        row = {
            "Concetto": c.name,
            "Famiglia": c.family,
            "Tipo": {"range": "soglia", "type": "tipo", "composite": "composto"}[c.kind],
            "Definizione": definition_str(c),
            "Feature": "sì" if c.is_feature else "no (ausiliario)",
            "Fondamento": c.why,
        }
        if concepts is not None:
            row["Copertura %"] = round(100.0 * concepts[c.name].mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def catalog_markdown(concepts: Optional[pd.DataFrame] = None) -> str:
    """Tabella Markdown del catalogo (usata nella relazione)."""
    cat = concept_catalog(concepts)
    cols = ["Concetto", "Definizione"] + (["Copertura %"] if concepts is not None else []) + ["Fondamento"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in cat.iterrows():
        cells = [f"`{r['Concetto']}`", r["Definizione"].replace("|", "\\|")]
        if concepts is not None:
            cells.append(f"{r['Copertura %']:.1f}")
        cells.append(r["Fondamento"].replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
