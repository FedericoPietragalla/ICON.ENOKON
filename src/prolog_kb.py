"""
Interrogazione della Knowledge Base Prolog (kb/wine_kb.pl).

Motori disponibili
  * "swipl"  : se l'eseguibile SWI-Prolog è nel PATH, i fatti dei vini vengono scritti in un
               file temporaneo insieme alla KB e interrogati con `classifica/2` (la KB gira
               davvero in Prolog).
  * "python" : motore di regole equivalente (src/knowledge.apply_rules), usato quando
               SWI-Prolog non c'è o l'esecuzione fallisce.
  * "auto"   : prova swipl e ricade su python.

Quando gira swipl, il risultato viene confrontato con quello del motore Python: la
percentuale di accordo (attesa 100%) è un controllo di coerenza tra le due implementazioni.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from src.knowledge import PL_TIPO_PRED, PROPERTIES, RULES, apply_rules

_MAIN = """
main :-
    forall(wine(V),
           ( classifica(V, C), format("~w ~w~n", [V, C]) )).
"""


def facts_text(df: pd.DataFrame) -> str:
    """Fatti Prolog per tutti i vini (raggruppati per predicato, quindi contigui)."""
    n = len(df)
    lines = [f"wine(w{i})." for i in range(n)]
    tipi = df[cfg.TIPO_COLUMN].to_numpy()
    lines += [f"{PL_TIPO_PRED}(w{i}, {tipi[i]})." for i in range(n)]
    for p in PROPERTIES:
        vals = df[p.column].to_numpy(dtype=float)
        lines += [f"{p.pl}(w{i}, {vals[i]:.6f})." for i in range(n)]
    return "\n".join(lines) + "\n"


class WineKB:
    def __init__(self, kb_path=cfg.KB_PATH, engine: str = "auto"):
        if engine not in ("auto", "swipl", "python"):
            raise ValueError("engine deve essere 'auto', 'swipl' o 'python'")
        self.kb_path = Path(kb_path)
        self.engine_requested = engine
        self.engine_used: Optional[str] = None
        self.agreement: Optional[float] = None       # accordo swipl vs python
        self.warning: Optional[str] = None
        self.fired_: Optional[np.ndarray] = None     # prima regola scattata (indice, -1 = default)

    # ------------------------------------------------------------------ swipl
    def _run_swipl(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        exe = shutil.which("swipl")
        if exe is None:
            self.warning = "SWI-Prolog (swipl) non trovato nel PATH"
            return None
        if not self.kb_path.exists():
            self.warning = f"KB non trovata: {self.kb_path} (esegui python kb/build_kb.py)"
            return None
        program = self.kb_path.read_text(encoding="utf-8") + "\n" + facts_text(df) + _MAIN
        try:
            with tempfile.TemporaryDirectory() as tmp:
                f = Path(tmp) / "enokon_run.pl"
                f.write_text(program, encoding="utf-8")
                proc = subprocess.run([exe, "-q", "-g", "main", "-t", "halt", str(f)],
                                      capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.warning = f"esecuzione di swipl fallita: {exc}"
            return None

        pred = np.full(len(df), -1, dtype=int)
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].startswith("w") and parts[1] in cfg.CLASS_NAMES:
                try:
                    pred[int(parts[0][1:])] = cfg.CLASS_NAMES.index(parts[1])
                except (ValueError, IndexError):
                    pass
        if proc.returncode != 0 or (pred < 0).any():
            self.warning = ("swipl ha restituito un risultato incompleto: "
                            + (proc.stderr.strip()[:200] or f"codice {proc.returncode}"))
            return None
        return pred

    # --------------------------------------------------------------- interfaccia
    def predict(self, df: pd.DataFrame, concepts: pd.DataFrame) -> np.ndarray:
        """Classe predetta (indice) per ogni vino di `df`; `concepts` = build_concepts(df)."""
        py_pred, self.fired_ = apply_rules(concepts)
        if self.engine_requested in ("auto", "swipl"):
            pl_pred = self._run_swipl(df)
            if pl_pred is not None:
                self.engine_used = "swipl"
                self.agreement = float((pl_pred == py_pred).mean())
                return pl_pred
            if self.engine_requested == "swipl":
                raise RuntimeError(f"Motore swipl richiesto ma non utilizzabile: {self.warning}")
        self.engine_used = "python"
        return py_pred

    def rule_report(self, y: np.ndarray) -> pd.DataFrame:
        """Per ogni regola: quanti vini decide (prima regola scattata) e con quale precisione."""
        if self.fired_ is None:
            raise RuntimeError("chiamare predict() prima di rule_report()")
        rows = []
        for i, r in enumerate(RULES):
            m = self.fired_ == i
            tgt = cfg.CLASS_NAMES.index(r.target)
            rows.append({"regola": r.name, "classe": r.target, "vini_decisi": int(m.sum()),
                         "copertura_%": round(100.0 * m.mean(), 1),
                         "precisione_%": round(100.0 * float((y[m] == tgt).mean()), 1) if m.any() else np.nan})
        m = self.fired_ < 0
        tgt = cfg.CLASS_NAMES.index("media")
        rows.append({"regola": "(default)", "classe": "media", "vini_decisi": int(m.sum()),
                     "copertura_%": round(100.0 * m.mean(), 1),
                     "precisione_%": round(100.0 * float((y[m] == tgt).mean()), 1) if m.any() else np.nan})
        return pd.DataFrame(rows)
