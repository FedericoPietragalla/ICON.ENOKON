"""
ENOKON - Conoscenza di background (BK) sul dominio enologico.

Questo modulo è l'UNICA fonte di verità della conoscenza di dominio. Da qui
vengono derivati, senza duplicazioni a mano:

  * le feature binarie "OWL" usate dai modelli      -> src/ontology_features.py
  * l'ontologia OWL (wine.owl)                      -> ontology/build_ontology.py
  * la Knowledge Base Prolog (wine_kb.pl)           -> kb/build_kb.py
  * il motore di regole in Python (fallback Prolog) -> src/prolog_kb.py

Per cambiare una soglia o aggiungere un concetto si modifica SOLO questo file e
si rigenerano ontologia e KB:

    python ontology/build_ontology.py && python kb/build_kb.py

Convenzione sugli intervalli: [lo, hi) salvo diversa indicazione (lo_incl / hi_incl).

Livelli della conoscenza
------------------------
1. Concetti di soglia (RANGE): discretizzano una misura chimica in un livello
   qualitativo ("alcol alto", "pH basso", ...), come farebbe un enologo.
2. Concetti composti (COMPOSITE): combinano concetti con AND/OR e rappresentano
   nozioni enologiche di livello superiore ("vino difettoso", "vino equilibrato").
3. Regole di classificazione (RULES): regole da sistema esperto che assegnano la
   classe di qualità a partire dai concetti.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import CLASS_NAMES, TIPO_COLUMN

# ----------------------------------------------------------------------------
# Proprietà (data property OWL e predicati Prolog dei fatti)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Prop:
    column: str   # colonna del DataFrame
    owl: str      # nome della data property OWL
    pl: str       # predicato Prolog (fatto  pl(VinoId, Valore))
    unit: str     # unità di misura
    label: str    # etichetta leggibile


PROPERTIES: Tuple[Prop, ...] = (
    Prop("fixed_acidity", "aciditaFissa", "acidita_fissa", "g/L", "acidità fissa"),
    Prop("volatile_acidity", "aciditaVolatile", "acidita_volatile", "g/L", "acidità volatile"),
    Prop("citric_acid", "acidoCitrico", "acido_citrico", "g/L", "acido citrico"),
    Prop("residual_sugar", "zuccheriResidui", "zuccheri_residui", "g/L", "zuccheri residui"),
    Prop("chlorides", "cloruri", "cloruri", "g/L", "cloruri"),
    Prop("free_sulfur_dioxide", "so2Libera", "so2_libera", "mg/L", "SO2 libera"),
    Prop("total_sulfur_dioxide", "so2Totale", "so2_totale", "mg/L", "SO2 totale"),
    Prop("density", "densita", "densita", "g/cm3", "densità"),
    Prop("pH", "ph", "ph", "", "pH"),
    Prop("sulphates", "solfati", "solfati", "g/L", "solfati"),
    Prop("alcohol", "gradoAlcolico", "grado_alcolico", "% vol", "grado alcolico"),
)
PROP_BY_COLUMN: Dict[str, Prop] = {p.column: p for p in PROPERTIES}

# Proprietà non numeriche / di sola ontologia
OWL_TIPO_PROP = "tipoVino"           # xsd:string  ('rosso' | 'bianco')
PL_TIPO_PRED = "tipo"
OWL_QUALITY_PROP = "punteggioQualita"  # xsd:integer (punteggio sensoriale)

# ----------------------------------------------------------------------------
# Concetti
# ----------------------------------------------------------------------------
RANGE, TYPE, COMPOSITE = "range", "type", "composite"

Expr = Union[str, Tuple[str, Tuple["Expr", ...]]]   # str = riferimento a un concetto


def AND(*args: Expr) -> Expr:
    return ("and", tuple(args))


def OR(*args: Expr) -> Expr:
    return ("or", tuple(args))


@dataclass(frozen=True)
class Concept:
    name: str                      # nome della classe OWL (CamelCase)
    pl: str                        # nome del predicato Prolog (snake_case)
    family: str                    # classe-famiglia OWL (superclasse)
    kind: str                      # RANGE | TYPE | COMPOSITE
    description: str
    why: str = ""                  # fondamento della soglia / della definizione
    column: Optional[str] = None   # RANGE: colonna del DataFrame
    lo: Optional[float] = None
    hi: Optional[float] = None
    lo_incl: bool = True
    hi_incl: bool = False
    value: Optional[str] = None    # TYPE: valore atteso di 'tipo'
    expr: Optional[Expr] = None    # COMPOSITE: espressione AND/OR
    is_feature: bool = True        # usato come feature binaria dai modelli?
    disjoint: bool = True          # mutuamente esclusivo con gli altri della famiglia?


def _rng(name, pl, family, column, desc, why="", lo=None, hi=None,
         lo_incl=True, hi_incl=False, feature=True, disjoint=True) -> Concept:
    return Concept(name, pl, family, RANGE, desc, why, column=column, lo=lo, hi=hi,
                   lo_incl=lo_incl, hi_incl=hi_incl, is_feature=feature, disjoint=disjoint)


def _typ(name, pl, value, desc, feature=True) -> Concept:
    return Concept(name, pl, "TipoVino", TYPE, desc, "Tipologia indicata nel dataset.",
                   value=value, is_feature=feature)


def _cmp(name, pl, expr, desc, why="", feature=True) -> Concept:
    return Concept(name, pl, "ProfiloEnologico", COMPOSITE, desc, why, expr=expr,
                   is_feature=feature, disjoint=False)


# NB: l'ordine conta: un concetto composto può riferirsi solo a concetti già definiti.
CONCEPTS: Tuple[Concept, ...] = (
    # ---- Tipo ---------------------------------------------------------------
    _typ("VinoRosso", "vino_rosso", "rosso", "Vino rosso."),
    _typ("VinoBianco", "vino_bianco", "bianco", "Vino bianco.", feature=False),

    # ---- Acidità fissa ------------------------------------------------------
    _rng("AciditaFissaBassa", "acidita_fissa_bassa", "ProfiloAciditaFissa", "fixed_acidity",
         "Vino poco acido.", "Soglia empirica: meno freschezza e minore stabilità.", hi=6.0),
    _rng("AciditaFissaAlta", "acidita_fissa_alta", "ProfiloAciditaFissa", "fixed_acidity",
         "Vino molto acido.", "Soglia empirica.", lo=8.5),

    # ---- Acidità volatile ---------------------------------------------------
    _rng("AciditaVolatileBassa", "acidita_volatile_bassa", "ProfiloAciditaVolatile", "volatile_acidity",
         "Fermentazione e conservazione pulite.",
         "Acido acetico ben al di sotto della soglia di percezione.", hi=0.30),
    _rng("AciditaVolatileElevata", "acidita_volatile_elevata", "ProfiloAciditaVolatile", "volatile_acidity",
         "Spunto acetico (difetto).",
         "Oltre ~0.6 g/L l'acido acetico diventa percepibile (nota di aceto/solvente): "
         "soglia sensoriale di uso comune, inferiore al limite di legge.", lo=0.60),

    # ---- Acido citrico ------------------------------------------------------
    _rng("AcidoCitricoBasso", "acido_citrico_basso", "ProfiloAcidoCitrico", "citric_acid",
         "Poco acido citrico.", "Soglia empirica: minore freschezza aromatica.", hi=0.15),
    _rng("AcidoCitricoElevato", "acido_citrico_elevato", "ProfiloAcidoCitrico", "citric_acid",
         "Molto acido citrico.", "Soglia empirica.", lo=0.50),

    # ---- Zuccheri residui (indicazioni di dolcezza UE per i vini fermi) ------
    _rng("Secco", "secco", "ProfiloZuccherino", "residual_sugar",
         "Vino secco.",
         "Indicazione di dolcezza UE: secco fino a 4 g/L (versione semplificata, "
         "senza la compensazione con l'acidità).", hi=4.0, hi_incl=True),
    _rng("Abboccato", "abboccato", "ProfiloZuccherino", "residual_sugar",
         "Vino abboccato (semisecco).", "Da 4 a 12 g/L (indicazione di dolcezza UE).",
         lo=4.0, lo_incl=False, hi=12.0, hi_incl=True),
    _rng("Amabile", "amabile", "ProfiloZuccherino", "residual_sugar",
         "Vino amabile o dolce.",
         "Oltre 12 g/L: le classi UE 'amabile' (12-45) e 'dolce' (>45) sono unite "
         "perché i vini dolci sono poco rappresentati nel dataset.", lo=12.0, lo_incl=False),

    # ---- Cloruri ------------------------------------------------------------
    _rng("CloruriBassi", "cloruri_bassi", "ProfiloCloruri", "chlorides",
         "Basso contenuto salino.", "Soglia empirica sul contenuto salino.", hi=0.05),
    _rng("CloruriElevati", "cloruri_elevati", "ProfiloCloruri", "chlorides",
         "Alto contenuto salino.", "Soglia empirica sul contenuto salino.", lo=0.10),

    # ---- Anidride solforosa libera -------------------------------------------
    _rng("SO2LiberaBassa", "so2_libera_bassa", "ProfiloSO2Libera", "free_sulfur_dioxide",
         "SO2 libera insufficiente.",
         "Protezione scarsa da ossidazione e microrganismi (soglia di uso comune).", hi=15.0),
    _rng("SO2LiberaAdeguata", "so2_libera_adeguata", "ProfiloSO2Libera", "free_sulfur_dioxide",
         "SO2 libera nell'intervallo di lavoro abituale.",
         "Protezione adeguata senza note pungenti.", lo=15.0, hi=50.0, hi_incl=True),
    _rng("SO2LiberaAlta", "so2_libera_alta", "ProfiloSO2Libera", "free_sulfur_dioxide",
         "SO2 libera eccessiva.",
         "Oltre ~50 mg/L la solforosa libera è percepibile all'olfatto (odore pungente).",
         lo=50.0, lo_incl=False),

    # ---- Anidride solforosa totale: classi ausiliarie per il limite di legge -----
    _rng("SO2TotaleOltre150", "so2_totale_oltre_150", "ProfiloSO2Totale", "total_sulfur_dioxide",
         "SO2 totale sopra 150 mg/L.", "Ausiliaria: limite UE per i vini rossi secchi.",
         lo=150.0, lo_incl=False, feature=False, disjoint=False),
    _rng("SO2TotaleOltre200", "so2_totale_oltre_200", "ProfiloSO2Totale", "total_sulfur_dioxide",
         "SO2 totale sopra 200 mg/L.", "Ausiliaria: limite UE per i vini bianchi secchi.",
         lo=200.0, lo_incl=False, feature=False, disjoint=False),
    _cmp("SO2TotaleEccessiva", "so2_totale_eccessiva",
         OR(AND("VinoRosso", "SO2TotaleOltre150"), AND("VinoBianco", "SO2TotaleOltre200")),
         "SO2 totale oltre il limite di legge per la tipologia.",
         "Limiti UE sull'anidride solforosa totale per vini secchi: 150 mg/L (rossi) e "
         "200 mg/L (bianchi). Versione semplificata: ignora i limiti più alti previsti "
         "per i vini con zuccheri residui >= 5 g/L."),

    # ---- Densità ------------------------------------------------------------
    _rng("DensitaBassa", "densita_bassa", "ProfiloDensita", "density",
         "Densità bassa: vino secco e alcolico.", "L'alcol abbassa la densità. Soglia empirica.",
         hi=0.993),
    _rng("DensitaAlta", "densita_alta", "ProfiloDensita", "density",
         "Densità alta: molto estratto e/o zuccheri.", "Soglia empirica.", lo=0.997),

    # ---- pH -----------------------------------------------------------------
    _rng("PHBasso", "ph_basso", "ProfiloPH", "pH",
         "pH basso: acidità elevata, vino fresco e stabile.", "Convenzione enologica.", hi=3.2),
    _rng("PHMedio", "ph_medio", "ProfiloPH", "pH",
         "pH nell'intervallo usuale.", "Convenzione enologica.", lo=3.2, hi=3.5),
    _rng("PHAlto", "ph_alto", "ProfiloPH", "pH",
         "pH alto: vino piatto, a rischio di instabilità microbiologica.",
         "Con pH alto la solforosa è meno efficace. Convenzione enologica.", lo=3.5),

    # ---- Solfati ------------------------------------------------------------
    _rng("SolfatiBassi", "solfati_bassi", "ProfiloSolfati", "sulphates",
         "Pochi solfati.", "Soglia empirica (additivo con azione antimicrobica/antiossidante).",
         hi=0.50),
    _rng("SolfatiAlti", "solfati_alti", "ProfiloSolfati", "sulphates",
         "Molti solfati.", "Soglia empirica.", lo=0.75),

    # ---- Grado alcolico -----------------------------------------------------
    _rng("AlcolBasso", "alcol_basso", "ProfiloAlcolico", "alcohol",
         "Vino leggero.", "Convenzione: sotto 10 % vol.", hi=10.0),
    _rng("AlcolMedio", "alcol_medio", "ProfiloAlcolico", "alcohol",
         "Vino di corpo medio.", "Convenzione: tra 10 e 12 % vol.", lo=10.0, hi=12.0),
    _rng("AlcolAlto", "alcol_alto", "ProfiloAlcolico", "alcohol",
         "Vino alcolico.", "Convenzione: da 12 % vol, indice di uve molto mature.", lo=12.0),

    # ---- Concetti composti (nozioni enologiche di livello superiore) ----------
    _cmp("RischioInstabilita", "rischio_instabilita", AND("PHAlto", "SO2LiberaBassa"),
         "Rischio di instabilità microbiologica/ossidativa.",
         "Un pH alto con SO2 libera bassa lascia il vino poco protetto "
         "(l'efficacia antimicrobica della solforosa cala al crescere del pH)."),
    _cmp("VinoDifettoso", "vino_difettoso",
         OR("AciditaVolatileElevata", "SO2TotaleEccessiva", "RischioInstabilita"),
         "Vino con almeno un difetto analitico.",
         "Spunto acetico, eccesso di solforosa oppure rischio di instabilità."),
    _cmp("VinoFresco", "vino_fresco",
         AND("PHBasso", OR("AlcolBasso", "AlcolMedio"), "AciditaVolatileBassa"),
         "Vino fresco e leggero.",
         "Alta acidità (pH basso), alcol contenuto e assenza di spunto acetico: "
         "profilo tipico dei vini bianchi giovani e del vinho verde."),
    _cmp("VinoEquilibrato", "vino_equilibrato",
         AND("AlcolMedio", "AciditaVolatileBassa", "SO2LiberaAdeguata", OR("PHBasso", "PHMedio")),
         "Vino equilibrato.",
         "Alcol medio, acidità volatile bassa, protezione solforosa adeguata e pH usuale."),
    _cmp("VinoMaturoEPulito", "vino_maturo_e_pulito", AND("AlcolAlto", "AciditaVolatileBassa"),
         "Vino da uve mature e ben vinificato.",
         "Grado alcolico alto (maturità delle uve) senza spunto acetico."),
)

CONCEPT_BY_NAME: Dict[str, Concept] = {c.name: c for c in CONCEPTS}
FEATURE_CONCEPTS: List[str] = [c.name for c in CONCEPTS if c.is_feature]
FAMILIES: List[str] = list(dict.fromkeys(c.family for c in CONCEPTS))


def disjoint_groups() -> Dict[str, List[str]]:
    """Famiglie di concetti mutuamente esclusivi (per gli assiomi AllDisjoint)."""
    groups: Dict[str, List[str]] = {}
    for c in CONCEPTS:
        if c.disjoint:
            groups.setdefault(c.family, []).append(c.name)
    return {f: names for f, names in groups.items() if len(names) > 1}


# ----------------------------------------------------------------------------
# Regole di classificazione (sistema esperto)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    name: str                 # identificativo (usato anche in Prolog come motivo)
    target: str               # classe assegnata: una voce di CLASS_NAMES
    body: Tuple[str, ...]     # congiunzione di concetti
    description: str


# Le regole di una classe a priorità più alta vengono provate per prime.
RULE_PRIORITY: Tuple[str, ...] = ("bassa", "alta")   # poi il default
DEFAULT_CLASS = "media"

_RULES = (
    Rule("difetto_enologico", "bassa", ("VinoDifettoso",),
         "Un vino con un difetto analitico (spunto acetico, eccesso di SO2, rischio di "
         "instabilità) è di qualità bassa."),
    Rule("poco_alcol_pochi_solfati", "bassa", ("AlcolBasso", "SolfatiBassi"),
         "Un vino povero di alcol e di solfati ha poca struttura e poca protezione: qualità bassa."),
    Rule("maturo_e_pulito", "alta", ("VinoMaturoEPulito",),
         "Un vino da uve mature e senza spunto acetico è di qualità alta."),
    Rule("equilibrato_con_solfati_alti", "alta", ("VinoEquilibrato", "SolfatiAlti"),
         "Un vino equilibrato con solfati alti è di qualità alta."),
)
RULES: Tuple[Rule, ...] = tuple(sorted(_RULES, key=lambda r: RULE_PRIORITY.index(r.target)))


# ----------------------------------------------------------------------------
# Valutazione in Python (stessa semantica di OWL e Prolog)
# ----------------------------------------------------------------------------


def _eval_expr(expr: Expr, values: Dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(expr, str):
        return values[expr]
    op, args = expr
    arrays = [_eval_expr(a, values) for a in args]
    return np.logical_and.reduce(arrays) if op == "and" else np.logical_or.reduce(arrays)


def evaluate_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """Valuta TUTTI i concetti su un DataFrame (colonne RAW_FEATURES + 'tipo').

    Restituisce un DataFrame booleano con una colonna per concetto, nell'ordine di
    definizione. Non ha parametri da stimare: è una funzione deterministica delle
    misure, quindi calcolarla su tutto il dataset non causa data leakage.
    """
    values: Dict[str, np.ndarray] = {}
    n = len(df)
    for c in CONCEPTS:
        if c.kind == RANGE:
            x = df[c.column].to_numpy(dtype=float)
            m = np.ones(n, dtype=bool)
            if c.lo is not None:
                m &= (x >= c.lo) if c.lo_incl else (x > c.lo)
            if c.hi is not None:
                m &= (x <= c.hi) if c.hi_incl else (x < c.hi)
        elif c.kind == TYPE:
            m = np.asarray(df[TIPO_COLUMN].to_numpy() == c.value)
        else:
            m = _eval_expr(c.expr, values)
        values[c.name] = np.asarray(m, dtype=bool)
    return pd.DataFrame(values, index=df.index)


def apply_rules(concepts: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Applica le regole di classificazione.

    Restituisce (classe_predetta, indice_regola): la classe è l'indice in CLASS_NAMES,
    l'indice di regola è la PRIMA regola scattata in ordine di priorità (-1 = default).
    """
    n = len(concepts)
    fired = np.full(n, -1, dtype=int)
    for i, rule in enumerate(RULES):
        m = np.logical_and.reduce([concepts[b].to_numpy(dtype=bool) for b in rule.body])
        fired = np.where((fired < 0) & m, i, fired)
    targets = np.array([CLASS_NAMES.index(r.target) for r in RULES] + [CLASS_NAMES.index(DEFAULT_CLASS)])
    pred = targets[np.where(fired >= 0, fired, len(RULES))]
    return pred.astype(int), fired


# ----------------------------------------------------------------------------
# Utilità di descrizione (documentazione, OWL, Prolog)
# ----------------------------------------------------------------------------


def to_dnf(expr: Expr) -> List[List[str]]:
    """Forma normale disgiuntiva: lista di congiunzioni (liste di nomi di concetto)."""
    if isinstance(expr, str):
        return [[expr]]
    op, args = expr
    parts = [to_dnf(a) for a in args]
    if op == "or":
        return [conj for p in parts for conj in p]
    result: List[List[str]] = [[]]
    for p in parts:
        result = [r + conj for r in result for conj in p]
    return [list(dict.fromkeys(conj)) for conj in result]


def _num(x: float) -> str:
    return repr(float(x))


def interval_str(c: Concept) -> str:
    lb = "[" if (c.lo is not None and c.lo_incl) else "("
    rb = "]" if (c.hi is not None and c.hi_incl) else ")"
    lo = "-∞" if c.lo is None else _num(c.lo)
    hi = "+∞" if c.hi is None else _num(c.hi)
    return f"{lb}{lo}, {hi}{rb}"


def expr_str(expr: Expr, top: bool = True) -> str:
    if isinstance(expr, str):
        return expr
    op, args = expr
    sep = " ∧ " if op == "and" else " ∨ "
    s = sep.join(expr_str(a, top=False) for a in args)
    return s if top else f"({s})"


def definition_str(c: Concept) -> str:
    """Definizione formale leggibile del concetto."""
    if c.kind == RANGE:
        p = PROP_BY_COLUMN[c.column]
        unit = f" {p.unit}" if p.unit else ""
        return f"{p.label} ∈ {interval_str(c)}{unit}"
    if c.kind == TYPE:
        return f'{OWL_TIPO_PROP} = "{c.value}"'
    return expr_str(c.expr)


def _self_check() -> None:
    """Controlli di coerenza eseguiti all'import (falliscono subito se la BK è incoerente)."""
    seen = set()
    for c in CONCEPTS:
        assert c.name not in seen, f"concetto duplicato: {c.name}"
        if c.kind == COMPOSITE:
            for conj in to_dnf(c.expr):
                for ref in conj:
                    assert ref in seen, f"{c.name} usa '{ref}' prima che sia definito"
        seen.add(c.name)
    assert len({c.pl for c in CONCEPTS}) == len(CONCEPTS), "nomi Prolog duplicati"
    for r in RULES:
        assert r.target in CLASS_NAMES, f"classe sconosciuta nella regola {r.name}"
        assert all(b in CONCEPT_BY_NAME for b in r.body), f"concetto sconosciuto in {r.name}"


_self_check()
