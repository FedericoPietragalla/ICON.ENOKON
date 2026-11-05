"""
Preprocessing del dataset Wine Quality.

  * caricamento e validazione del CSV (data/wine_quality.csv)
  * rimozione dei record duplicati
  * discretizzazione del punteggio sensoriale nelle 3 classi bersaglio
  * split stratificati ripetuti e sottoinsieme etichettato per il semi-supervisionato
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

import config as cfg


def quality_to_class(quality) -> np.ndarray:
    """Punteggio sensoriale -> indice di classe (0 = bassa, 1 = media, 2 = alta)."""
    q = np.asarray(quality)
    return np.where(q <= cfg.LOW_MAX, 0, np.where(q >= cfg.HIGH_MIN, 2, 1)).astype(int)


def load_dataset(path=cfg.DATA_PATH, drop_duplicates: bool = cfg.DROP_DUPLICATES) -> pd.DataFrame:
    """Carica e prepara il dataset.

    Colonne aggiunte: 'tipo' ('rosso'/'bianco'), 'is_red' (0/1), 'quality_class' (0/1/2),
    'quality_label' (nome della classe). Le statistiche di pulizia sono in df.attrs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {path}\n"
            "Scaricalo con:  python data/download_dataset.py"
        )
    df = pd.read_csv(path)
    needed = cfg.RAW_FEATURES + [cfg.TYPE_COLUMN, cfg.QUALITY_COLUMN]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    df = df[needed].dropna()
    n_loaded = len(df)
    if drop_duplicates:
        df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df[cfg.TYPE_COLUMN] = df[cfg.TYPE_COLUMN].astype(str).str.strip().str.lower()
    unknown = set(df[cfg.TYPE_COLUMN].unique()) - {"red", "white"}
    if unknown:
        raise ValueError(f"Valori non ammessi in '{cfg.TYPE_COLUMN}': {unknown}")
    df[cfg.TIPO_COLUMN] = df[cfg.TYPE_COLUMN].map({"red": "rosso", "white": "bianco"})
    df["is_red"] = (df[cfg.TYPE_COLUMN] == "red").astype(int)
    df["quality_class"] = quality_to_class(df[cfg.QUALITY_COLUMN].to_numpy())
    df["quality_label"] = df["quality_class"].map(dict(enumerate(cfg.CLASS_NAMES)))

    df.attrs["n_loaded"] = int(n_loaded)
    df.attrs["n_duplicates"] = int(n_loaded - len(df)) if drop_duplicates else 0
    return df


def raw_feature_names() -> List[str]:
    return list(cfg.RAW_FEATURES) + ["is_red"]


def raw_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Misure chimiche + indicatore di tipo (12 colonne numeriche)."""
    return df[raw_feature_names()].to_numpy(dtype=float)


def dataset_summary(df: pd.DataFrame) -> dict:
    counts = df["quality_class"].value_counts().sort_index()
    return {
        "n_loaded": df.attrs.get("n_loaded", len(df)),
        "n_duplicates": df.attrs.get("n_duplicates", 0),
        "n": len(df),
        "n_red": int((df[cfg.TYPE_COLUMN] == "red").sum()),
        "n_white": int((df[cfg.TYPE_COLUMN] == "white").sum()),
        "class_counts": {cfg.CLASS_NAMES[i]: int(counts.get(i, 0)) for i in range(len(cfg.CLASS_NAMES))},
    }


def make_splits(y: np.ndarray, n_runs: int, test_size: float, seed: int
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """`n_runs` split stratificati indipendenti (train_idx, test_idx)."""
    sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=test_size, random_state=seed)
    return [(tr, te) for tr, te in sss.split(np.zeros(len(y)), y)]


def labeled_mask(y_train: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Maschera booleana degli esempi di training che mantengono l'etichetta.

    Il resto viene trattato come non etichettato (si ignora la sua etichetta). Il
    campionamento è stratificato, così ogni classe ha almeno qualche esempio.
    """
    n = len(y_train)
    mask = np.zeros(n, dtype=bool)
    if fraction >= 1.0:
        mask[:] = True
        return mask
    k = max(int(round(fraction * n)), len(np.unique(y_train)) * 2)
    idx, _ = train_test_split(np.arange(n), train_size=min(k, n - 1), stratify=y_train,
                              random_state=seed)
    mask[idx] = True
    return mask
