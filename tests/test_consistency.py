"""
Test di coerenza e correttezza di ENOKON.

    python -m pytest tests -q          # se pytest è installato
    python tests/test_consistency.py   # senza pytest

Verificano che
  * l'ontologia OWL generata (wine.owl) esprima ESATTAMENTE le definizioni di src/knowledge.py;
  * la KB Prolog generata (wine_kb.pl) derivi gli stessi concetti e la stessa classe di qualità;
  * i concetti mutuamente esclusivi lo siano davvero (anche sui valori esattamente sulle soglie);
  * i modelli probabilistici siano corretti (BNB = scikit-learn, EM monotono, TAN sensato);
  * il p-value di Nemenyi riproduca i valori critici di Demsar (2006).

I test OWL e Prolog non usano reasoner né SWI-Prolog: contengono due piccoli interpreti
indipendenti che leggono SOLO i file generati, così controllano il generatore e non solo la
logica Python.
"""
from __future__ import annotations

import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from kb import build_kb  # noqa: E402
from ontology import build_ontology  # noqa: E402
from src import knowledge as kn  # noqa: E402

RDF, RDFS, OWL = build_ontology.RDF, build_ontology.RDFS, build_ontology.OWL


# ----------------------------------------------------------------------------- dati di prova
def make_wines(n_random: int = 400, seed: int = 0) -> pd.DataFrame:
    """Vini casuali + vini con una misura ESATTAMENTE sulle soglie (e subito sotto/sopra)."""
    rng = np.random.default_rng(seed)
    base = {
        "fixed_acidity": rng.uniform(4, 15, n_random), "volatile_acidity": rng.uniform(0.1, 1.2, n_random),
        "citric_acid": rng.uniform(0, 1, n_random), "residual_sugar": rng.gamma(1.5, 4, n_random),
        "chlorides": rng.uniform(0.01, 0.2, n_random), "free_sulfur_dioxide": rng.uniform(1, 80, n_random),
        "total_sulfur_dioxide": rng.uniform(5, 300, n_random), "density": rng.uniform(0.989, 1.002, n_random),
        "pH": rng.uniform(2.8, 3.9, n_random), "sulphates": rng.uniform(0.25, 1.2, n_random),
        "alcohol": rng.uniform(8, 14.5, n_random),
        cfg.TIPO_COLUMN: rng.choice(["rosso", "bianco"], n_random),
    }
    df = pd.DataFrame(base)
    edge_rows = []
    for c in kn.CONCEPTS:
        if c.kind != kn.RANGE:
            continue
        for bound in (c.lo, c.hi):
            if bound is None:
                continue
            for v in (np.nextafter(bound, -np.inf), bound, np.nextafter(bound, np.inf)):
                for tipo in ("rosso", "bianco"):
                    row = df.iloc[rng.integers(0, n_random)].copy()
                    row[c.column] = v
                    row[cfg.TIPO_COLUMN] = tipo
                    edge_rows.append(row)
    return pd.concat([df, pd.DataFrame(edge_rows)], ignore_index=True)


# ----------------------------------------------------------------------------- interprete OWL
def _q(ns, tag):
    return f"{{{ns}}}{tag}"


class OwlEvaluator:
    """Valuta le classi definite leggendo SOLO wine.owl (restrizioni su data property)."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.defs = {}
        for cls in root.findall(_q(OWL, "Class")):
            about = cls.get(_q(RDF, "about"))
            eq = cls.find(_q(OWL, "equivalentClass"))
            if about and eq is not None:
                self.defs[about.split("#")[1]] = eq[0]

    def holds(self, name, ind):
        return self._eval(self.defs[name], ind)

    def _eval(self, el, ind):
        if el.tag == _q(OWL, "Class"):
            about = el.get(_q(RDF, "about"))
            if about:
                name = about.split("#")[1]
                return True if name == "Vino" else self.holds(name, ind)
            inter, union = el.find(_q(OWL, "intersectionOf")), el.find(_q(OWL, "unionOf"))
            if inter is not None:
                return all(self._eval(c, ind) for c in inter)
            if union is not None:
                return any(self._eval(c, ind) for c in union)
        if el.tag == _q(OWL, "Restriction"):
            prop = el.find(_q(OWL, "onProperty")).get(_q(RDF, "resource")).split("#")[1]
            has_value = el.find(_q(OWL, "hasValue"))
            if has_value is not None:
                return ind[prop] == has_value.text
            dt = el.find(_q(OWL, "someValuesFrom")).find(_q(RDFS, "Datatype"))
            x, ok = float(ind[prop]), True
            for desc in dt.find(_q(OWL, "withRestrictions")):
                for facet in desc:
                    kind, v = facet.tag.split("}")[1], float(facet.text)
                    ok &= {"minInclusive": x >= v, "minExclusive": x > v,
                           "maxInclusive": x <= v, "maxExclusive": x < v}[kind]
            return ok
        raise ValueError(f"costrutto OWL non gestito: {el.tag}")


def test_owl_matches_python():
    wines = make_wines()
    expected = kn.evaluate_concepts(wines)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wine.owl"
        build_ontology.build(path)
        owl = OwlEvaluator(path)
        for c in kn.CONCEPTS:
            got = np.array([owl.holds(c.name, {**{p.owl: row[p.column] for p in kn.PROPERTIES},
                                              kn.OWL_TIPO_PROP: row[cfg.TIPO_COLUMN]})
                            for _, row in wines.iterrows()])
            assert (got == expected[c.name].to_numpy()).all(), f"OWL e Python divergono su {c.name}"


def test_owl_is_well_formed_and_complete():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wine.owl"
        info = build_ontology.build(path)
        root = ET.parse(path).getroot()
        declared = {e.get(_q(RDF, "about")).split("#")[1] for e in root.findall(_q(OWL, "Class"))
                    if e.get(_q(RDF, "about"))}
        # ogni classe e proprietà citata deve essere dichiarata
        for el in root.iter():
            ref = el.get(_q(RDF, "about")) or el.get(_q(RDF, "resource"))
            if ref and ref.startswith(cfg.ONTOLOGY_IRI + "#"):
                name = ref.split("#")[1]
                props = {e.get(_q(RDF, "about")).split("#")[1] for e in root.findall(_q(OWL, "DatatypeProperty"))}
                assert name in declared | props, f"riferimento non dichiarato: {name}"
        assert info["classi_definite"] == len(kn.CONCEPTS) + len(cfg.CLASS_NAMES)
        assert info["data_property"] == len(kn.PROPERTIES) + 2


# --------------------------------------------------------------------------- interprete Prolog
def _split_goals(body: str):
    goals, depth, cur = [], 0, ""
    for ch in body:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            goals.append(cur.strip())
            cur = ""
        else:
            cur += ch
    goals.append(cur.strip())
    return goals


class MiniProlog:
    """Interprete minimale per le clausole generate da kb/build_kb.py."""

    def __init__(self, text):
        self.clauses = {}
        for line in text.splitlines():
            m = re.match(r"^(\w+)\(([^)]*)\) :- (.*)\.$", line.strip())
            if m:
                name, args, body = m.groups()
                if name == "classifica":
                    continue
                self.clauses.setdefault(name, []).append(([a.strip() for a in args.split(",")], _split_goals(body)))
        self.order = re.findall(r"regola_(\w+)\(V, M\) -> C = \w+", text)
        self.default = re.search(r"C = (\w+), Motivo = nessuna_regola", text).group(1)

    def _prove(self, goals, facts, env):
        if not goals:
            return True
        g, rest = goals[0], goals[1:]
        cmp_ = re.match(r"^(\w+)\s*(>=|=<|>|<)\s*([-\d.eE]+)$", g)
        if cmp_:
            var, op, num = cmp_.groups()
            x, v = env[var], float(num)
            ok = {">=": x >= v, "=<": x <= v, ">": x > v, "<": x < v}[op]
            return ok and self._prove(rest, facts, env)
        name, args = re.match(r"^(\w+)\((.*)\)$", g).groups()
        args = [a.strip() for a in args.split(",")]
        if name in self.clauses:                                  # regola: V soltanto
            return self.holds(name, facts) and self._prove(rest, facts, env)
        value = facts[name]                                       # fatto: name(V, X | atomo)
        if args[1][0].isupper():
            return self._prove(rest, facts, {**env, args[1]: value})
        return value == args[1] and self._prove(rest, facts, env)

    def holds(self, name, facts):
        return any(self._prove(body, facts, {}) for _, body in self.clauses[name])

    def classify(self, facts):
        for cls in self.order:
            if any(self._prove(body, facts, {}) for _, body in self.clauses[f"regola_{cls}"]):
                return cls
        return self.default


def test_prolog_matches_python():
    wines = make_wines(n_random=250, seed=3)
    concepts = kn.evaluate_concepts(wines)
    pred, _ = kn.apply_rules(concepts)
    pl = MiniProlog(build_kb.generate_prolog())
    for i, row in wines.iterrows():
        facts = {p.pl: float(row[p.column]) for p in kn.PROPERTIES}
        facts[kn.PL_TIPO_PRED] = row[cfg.TIPO_COLUMN]
        for c in kn.CONCEPTS:
            assert pl.holds(c.pl, facts) == bool(concepts.loc[i, c.name]), f"Prolog e Python divergono su {c.name}"
        assert pl.classify(facts) == cfg.CLASS_NAMES[pred[i]], "classe di qualità diversa tra Prolog e Python"


# ---------------------------------------------------------------------------- struttura della BK
def test_disjoint_concepts_are_disjoint():
    wines = make_wines()
    concepts = kn.evaluate_concepts(wines)
    for family, names in kn.disjoint_groups().items():
        assert (concepts[names].sum(axis=1) <= 1).all(), f"concetti sovrapposti nella famiglia {family}"


def test_rule_priority_and_default():
    concepts = pd.DataFrame({c.name: [False, False] for c in kn.CONCEPTS})
    pred, fired = kn.apply_rules(concepts)
    assert (pred == cfg.CLASS_NAMES.index(kn.DEFAULT_CLASS)).all() and (fired == -1).all()
    both = concepts.copy()
    for name in ("VinoDifettoso", "VinoMaturoEPulito"):          # difetto vince sulla regola "alta"
        both[name] = True
    assert kn.apply_rules(both)[0][0] == cfg.CLASS_NAMES.index("bassa")


# ---------------------------------------------------------------------------------- modelli
def _toy_data(seed=0, K=3, D=30):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.1, 0.9, (K, D))

    def sample(n):
        y = rng.integers(0, K, n)
        return (rng.random((n, D)) < theta[y]).astype(np.uint8), y
    return sample, K


def test_bnb_matches_sklearn():
    from sklearn.naive_bayes import BernoulliNB as SkNB
    from src.bernoulli_nb import BernoulliNB
    sample, K = _toy_data()
    X, y = sample(300)
    Xt, _ = sample(500)
    mine = BernoulliNB(K, 1.0).fit(X, y)
    sk = SkNB(alpha=1.0, class_prior=np.exp(mine.class_log_prior_)).fit(X, y)
    assert np.allclose(mine.log_theta_, sk.feature_log_prob_)
    assert (mine.predict(Xt) == sk.predict(Xt)).all()


def test_em_objective_is_monotone_and_helps_when_assumptions_hold():
    from scipy.special import logsumexp
    from src.bernoulli_nb import BernoulliNB
    from src.bnb_em import SemiSupervisedBernoulliNB
    sample, K = _toy_data(seed=1)
    Xl, yl = sample(30)
    Xu, _ = sample(3000)
    Xt, yt = sample(3000)

    def map_objective(m):        # log-verosimiglianza + prior di Laplace (alpha = 1)
        ll = m.log_joint(Xl)[np.arange(len(yl)), yl].sum() + logsumexp(m.log_joint(Xu), axis=1).sum()
        return ll + m.class_log_prior_.sum() + (m.log_theta_ + m.log_1m_theta_).sum()
    vals = [map_objective(SemiSupervisedBernoulliNB(K, 1.0, 1.0, it, 0.0).fit(Xl, yl, Xu)) for it in range(1, 12)]
    assert (np.diff(vals) > -1e-6).all(), "l'obiettivo EM deve essere non decrescente"
    nb = BernoulliNB(K).fit(Xl, yl)
    em = SemiSupervisedBernoulliNB(K, 1.0, 1.0, 100, 1e-6).fit(Xl, yl, Xu)
    assert (em.predict(Xt) == yt).mean() > (nb.predict(Xt) == yt).mean(), "con ipotesi NB vere l'EM deve aiutare"


def test_tan_links_dependent_features_and_is_a_tree():
    from src.bayesian_network import TreeAugmentedNaiveBayes
    rng = np.random.default_rng(2)
    n = 4000
    y = rng.integers(0, 2, n)
    x0 = (rng.random(n) < np.where(y == 1, 0.8, 0.2)).astype(np.uint8)
    x1 = np.where(rng.random(n) < 0.05, 1 - x0, x0).astype(np.uint8)          # quasi copia di x0
    x2 = (rng.random(n) < np.where(y == 1, 0.6, 0.4)).astype(np.uint8)        # indipendente da x0, x1 | y
    X = np.c_[x0, x1, x2]
    tan = TreeAugmentedNaiveBayes(2, 1.0, ["x0", "x1", "x2"]).fit(X, y)
    edges = {frozenset(e) for e in tan.edges()}
    assert frozenset(("x0", "x1")) in edges, "TAN deve collegare le due feature quasi identiche"
    assert len(edges) == 2 and (tan.parent_ < 0).sum() == 1                    # albero con una sola radice
    for j in range(3):                                                         # nessun ciclo
        seen, node = set(), j
        while node >= 0:
            assert node not in seen
            seen.add(node)
            node = tan.parent_[node]
    proba = tan.predict_proba(X[:50])
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_nemenyi_critical_values_match_demsar():
    from scipy import stats
    for k, expected in ((3, 2.343), (10, 3.164)):                              # Demsar 2006, tab. 5
        q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        assert abs(q - expected) < 0.002


def test_labeled_mask_is_stratified():
    from src import preprocessing as prep
    y = np.repeat([0, 1, 2], [500, 300, 200])
    mask = prep.labeled_mask(y, 0.1, seed=0)
    assert mask.sum() == 100
    assert all(abs((y[mask] == k).mean() - (y == k).mean()) < 0.02 for k in range(3))
    assert prep.labeled_mask(y, 1.0, seed=0).all()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:                  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} test superati")
    sys.exit(1 if failed else 0)
