"""
Genera la Knowledge Base Prolog  kb/wine_kb.pl  a partire da src/knowledge.py.

    python kb/build_kb.py

Contenuto del file generato
  1. dichiarazioni dei predicati dinamici dei fatti (uno per misura chimica)
  2. regole che derivano i concetti di soglia      es.  alcol_alto(V) :- grado_alcolico(V, X), X >= 12.0.
  3. regole che derivano i concetti composti       es.  rischio_instabilita(V) :- ph_alto(V), so2_libera_bassa(V).
  4. regole di classificazione + spiegazione       classifica(V, Classe, Motivo)

I fatti dei singoli vini NON sono nel file: li asserisce src/prolog_kb.py (oppure si
inseriscono a mano per provare la KB, vedi l'esempio in fondo al file generato).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from src.knowledge import (CONCEPTS, COMPOSITE, DEFAULT_CLASS, PL_TIPO_PRED, PROP_BY_COLUMN,  # noqa: E402
                           PROPERTIES, RANGE, RULE_PRIORITY, RULES, CONCEPT_BY_NAME,
                           definition_str, to_dnf)


def _num(x: float) -> str:
    return repr(float(x))


def _range_clause(c) -> str:
    p = PROP_BY_COLUMN[c.column]
    conds = []
    if c.lo is not None:
        conds.append(f"X {'>=' if c.lo_incl else '>'} {_num(c.lo)}")
    if c.hi is not None:
        conds.append(f"X {'=<' if c.hi_incl else '<'} {_num(c.hi)}")
    return f"{c.pl}(V) :- " + ", ".join([f"{p.pl}(V, X)"] + conds) + "."


def _concept_clauses(c) -> list[str]:
    if c.kind == RANGE:
        return [_range_clause(c)]
    if c.kind == COMPOSITE:
        return [f"{c.pl}(V) :- " + ", ".join(f"{CONCEPT_BY_NAME[r].pl}(V)" for r in conj) + "."
                for conj in to_dnf(c.expr)]
    return [f"{c.pl}(V) :- {PL_TIPO_PRED}(V, {c.value})."]


def generate_prolog() -> str:
    out: list[str] = []
    w = out.append
    w("%% =====================================================================")
    w("%%  ENOKON - Knowledge Base Prolog sul dominio enologico")
    w("%%  FILE GENERATO da kb/build_kb.py a partire da src/knowledge.py:")
    w("%%  per modificarlo cambia src/knowledge.py e rigenera (python kb/build_kb.py).")
    w("%% =====================================================================")
    w("")
    w(":- encoding(utf8).")
    w("")
    w("%% --- 1. Fatti attesi (uno per vino) -----------------------------------")
    w("%%   wine(Id).                        identificativo del vino")
    w(f"%%   {PL_TIPO_PRED}(Id, rosso | bianco).")
    for p in PROPERTIES:
        unit = f" [{p.unit}]" if p.unit else ""
        w(f"%%   {p.pl}(Id, Valore).".ljust(44) + f"% {p.label}{unit}")
    dyn = ["wine/1", f"{PL_TIPO_PRED}/2"] + [f"{p.pl}/2" for p in PROPERTIES]
    w("")
    w(":- dynamic " + ", ".join(dyn) + ".")
    w("")
    w("%% --- 2. Concetti di soglia e di tipo ----------------------------------")
    for c in CONCEPTS:
        if c.kind == COMPOSITE:
            continue
        w(f"%% {c.name}: {definition_str(c)}")
        for line in _concept_clauses(c):
            w(line)
    w("")
    w("%% --- 3. Concetti composti (nozioni enologiche) --------------------------")
    for c in CONCEPTS:
        if c.kind != COMPOSITE:
            continue
        w(f"%% {c.name} := {definition_str(c)}")
        for line in _concept_clauses(c):
            w(line)
    w("")
    w("%% --- 4. Regole di classificazione della qualità ------------------------")
    w("%% regola_<classe>(Vino, Motivo): le regole sono provate in ordine di priorità")
    w(f"%% ({' > '.join(RULE_PRIORITY)} > {DEFAULT_CLASS}: la prima che scatta decide).")
    for r in RULES:
        w(f"%% {r.description}")
        w(f"regola_{r.target}(V, {r.name}) :- " + ", ".join(f"{CONCEPT_BY_NAME[b].pl}(V)" for b in r.body) + ".")
    w("")
    w("%% classifica(+Vino, ?Classe, ?Motivo): classe di qualità e regola che l'ha decisa.")
    w("classifica(V, C, Motivo) :-")
    first = True
    for t in RULE_PRIORITY:
        w(f"    {'(   ' if first else ';   '}regola_{t}(V, M) -> C = {t}, Motivo = M")
        first = False
    w(f"    ;   C = {DEFAULT_CLASS}, Motivo = nessuna_regola")
    w("    ).")
    w("")
    w("classifica(V, C) :- classifica(V, C, _).")
    w("")
    w("%% --- Esempio d'uso (SWI-Prolog) -----------------------------------------")
    w("%%   ?- [wine_kb].")
    w("%%   ?- assertz(wine(v1)), assertz(tipo(v1, bianco)),")
    w("%%      assertz(grado_alcolico(v1, 12.5)), assertz(acidita_volatile(v1, 0.22)),")
    w("%%      assertz(acidita_fissa(v1, 6.8)), assertz(acido_citrico(v1, 0.3)),")
    w("%%      assertz(zuccheri_residui(v1, 2.0)), assertz(cloruri(v1, 0.04)),")
    w("%%      assertz(so2_libera(v1, 30.0)), assertz(so2_totale(v1, 110.0)),")
    w("%%      assertz(densita(v1, 0.991)), assertz(ph(v1, 3.2)), assertz(solfati(v1, 0.5)).")
    w("%%   ?- classifica(v1, Classe, Motivo).")
    w("%%   Classe = alta, Motivo = maturo_e_pulito.")
    return "\n".join(out) + "\n"


def build(path: Path = cfg.KB_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_prolog(), encoding="utf-8")
    return path


if __name__ == "__main__":
    p = build()
    print(f"KB Prolog scritta in {p}  ({len(CONCEPTS)} concetti, {len(RULES)} regole)")
