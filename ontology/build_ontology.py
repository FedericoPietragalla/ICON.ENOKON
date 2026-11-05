"""
Genera l'ontologia OWL 2  ontology/wine.owl  a partire da src/knowledge.py.

    python ontology/build_ontology.py            # genera il file
    python ontology/build_ontology.py --inspect  # riassunto (e, se installato, caricamento con owlready2)

Il file è RDF/XML standard e si apre con Protégé o owlready2. Non servono librerie esterne
per generarlo.

Struttura
  Vino                              classe radice: un campione di vino
   ├─ <Famiglia>  (es. ProfiloAlcolico, ProfiloPH, TipoVino, ProfiloEnologico, QualitaVino)
   │    └─ <Concetto>  classe DEFINITA: equivalentClass = Vino ⊓ (restrizione su data property)
   │                    o combinazione AND/OR di altri concetti
  Data property funzionali: una per misura chimica (xsd:decimal), tipoVino (xsd:string),
  punteggioQualita (xsd:integer).
  Assiomi AllDisjointClasses per i concetti mutuamente esclusivi di una famiglia.

Con un reasoner (HermiT/Pellet in Protégé) un individuo di tipo Vino con le sue misure viene
classificato automaticamente nei concetti che soddisfa, ottenendo le stesse feature calcolate
da src/ontology_features.py (la coerenza è verificata da tests/test_consistency.py).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from src.knowledge import (CONCEPTS, COMPOSITE, FAMILIES, OWL_QUALITY_PROP, OWL_TIPO_PROP,  # noqa: E402
                           PROPERTIES, PROP_BY_COLUMN, RANGE, TYPE, definition_str,
                           disjoint_groups)

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
XML = "http://www.w3.org/XML/1998/namespace"
for _prefix, _uri in (("rdf", RDF), ("rdfs", RDFS), ("owl", OWL), ("xsd", XSD)):
    ET.register_namespace(_prefix, _uri)

BASE = cfg.ONTOLOGY_IRI


def q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def iri(name: str) -> str:
    return f"{BASE}#{name}"


def _sub(parent, ns, tag, attrib=None, text=None):
    e = ET.SubElement(parent, q(ns, tag), attrib or {})
    if text is not None:
        e.text = text
    return e


def _label(parent, label, comment=None):
    _sub(parent, RDFS, "label", {q(XML, "lang"): "it"}, label)
    if comment:
        _sub(parent, RDFS, "comment", {q(XML, "lang"): "it"}, comment)


def _class_ref(parent, name):
    return _sub(parent, OWL, "Class", {q(RDF, "about"): iri(name)})


def _lex(x, datatype: str) -> str:
    return str(int(x)) if datatype == "integer" else repr(float(x))


def _data_restriction(parent, prop, datatype, lo, lo_incl, hi, hi_incl):
    """∃ prop . datatype[facce]   (restrizione di dominio sui valori della data property)"""
    r = _sub(parent, OWL, "Restriction")
    _sub(r, OWL, "onProperty", {q(RDF, "resource"): iri(prop)})
    svf = _sub(r, OWL, "someValuesFrom")
    dt = _sub(svf, RDFS, "Datatype")
    _sub(dt, OWL, "onDatatype", {q(RDF, "resource"): XSD + datatype})
    wr = _sub(dt, OWL, "withRestrictions", {q(RDF, "parseType"): "Collection"})
    for bound, incl, kind in ((lo, lo_incl, "min"), (hi, hi_incl, "max")):
        if bound is not None:
            facet = f"{kind}{'Inclusive' if incl else 'Exclusive'}"
            d = _sub(wr, RDF, "Description")
            _sub(d, XSD, facet, {q(RDF, "datatype"): XSD + datatype}, _lex(bound, datatype))
    return r


def _expr(parent, expr):
    """Espressione AND/OR di concetti -> intersectionOf / unionOf."""
    if isinstance(expr, str):
        return _class_ref(parent, expr)
    op, args = expr
    cls = _sub(parent, OWL, "Class")
    coll = _sub(cls, OWL, "intersectionOf" if op == "and" else "unionOf",
                {q(RDF, "parseType"): "Collection"})
    for a in args:
        _expr(coll, a)
    return cls


def _quality_classes():
    """(nome classe, lo, hi) sul punteggio intero; estremi inclusivi."""
    out = []
    for name in cfg.CLASS_NAMES:
        lo = {"bassa": None, "media": cfg.LOW_MAX + 1, "alta": cfg.HIGH_MIN}[name]
        hi = {"bassa": cfg.LOW_MAX, "media": cfg.HIGH_MIN - 1, "alta": None}[name]
        out.append((f"Qualita{name.capitalize()}", name, lo, hi))
    return out


def build_tree() -> ET.ElementTree:
    root = ET.Element(q(RDF, "RDF"))
    ont = _sub(root, OWL, "Ontology", {q(RDF, "about"): BASE})
    _label(ont, "ENOKON - ontologia del vino",
           "Conoscenza di background per la classificazione della qualità del vino "
           "(vinho verde, UCI Wine Quality). Generata da ontology/build_ontology.py.")

    # --- classe radice ------------------------------------------------------------
    vino = _sub(root, OWL, "Class", {q(RDF, "about"): iri("Vino")})
    _label(vino, "Vino", "Campione di vino descritto da misure chimiche e dal punteggio sensoriale.")

    # --- data property -------------------------------------------------------------
    def data_prop(name, label, datatype, comment=""):
        dp = _sub(root, OWL, "DatatypeProperty", {q(RDF, "about"): iri(name)})
        _sub(dp, RDF, "type", {q(RDF, "resource"): OWL + "FunctionalProperty"})
        _sub(dp, RDFS, "domain", {q(RDF, "resource"): iri("Vino")})
        _sub(dp, RDFS, "range", {q(RDF, "resource"): XSD + datatype})
        _label(dp, label, comment or None)

    for p in PROPERTIES:
        data_prop(p.owl, p.label, "decimal", f"Unità di misura: {p.unit}" if p.unit else "Adimensionale.")
    data_prop(OWL_TIPO_PROP, "tipo di vino", "string", "Valori ammessi: 'rosso', 'bianco'.")
    data_prop(OWL_QUALITY_PROP, "punteggio di qualità", "integer", "Punteggio sensoriale (mediana di 3+ degustatori).")

    # --- classi famiglia -----------------------------------------------------------
    for fam in FAMILIES + ["QualitaVino"]:
        c = _sub(root, OWL, "Class", {q(RDF, "about"): iri(fam)})
        _sub(c, RDFS, "subClassOf", {q(RDF, "resource"): iri("Vino")})
        _label(c, fam, "Classe di raggruppamento (superclasse dei concetti omonimi).")

    # --- concetti definiti ----------------------------------------------------------
    for c in CONCEPTS:
        cls = _sub(root, OWL, "Class", {q(RDF, "about"): iri(c.name)})
        _label(cls, c.description, f"{definition_str(c)}. {c.why}".strip())
        _sub(cls, RDFS, "subClassOf", {q(RDF, "resource"): iri(c.family)})
        eq = _sub(cls, OWL, "equivalentClass")
        if c.kind == COMPOSITE:
            _expr(eq, c.expr)
            continue
        inter = _sub(_sub(eq, OWL, "Class"), OWL, "intersectionOf", {q(RDF, "parseType"): "Collection"})
        _class_ref(inter, "Vino")
        if c.kind == RANGE:
            _data_restriction(inter, PROP_BY_COLUMN[c.column].owl, "decimal", c.lo, c.lo_incl, c.hi, c.hi_incl)
        elif c.kind == TYPE:
            r = _sub(inter, OWL, "Restriction")
            _sub(r, OWL, "onProperty", {q(RDF, "resource"): iri(OWL_TIPO_PROP)})
            _sub(r, OWL, "hasValue", {q(RDF, "datatype"): XSD + "string"}, c.value)

    # --- classi di qualità (bersaglio) ---------------------------------------------
    qnames = []
    for name, label, lo, hi in _quality_classes():
        cls = _sub(root, OWL, "Class", {q(RDF, "about"): iri(name)})
        _label(cls, f"Vino di qualità {label}", f"punteggioQualita ∈ [{lo if lo is not None else '-∞'}, "
                                                  f"{hi if hi is not None else '+∞'}]")
        _sub(cls, RDFS, "subClassOf", {q(RDF, "resource"): iri("QualitaVino")})
        eq = _sub(cls, OWL, "equivalentClass")
        inter = _sub(_sub(eq, OWL, "Class"), OWL, "intersectionOf", {q(RDF, "parseType"): "Collection"})
        _class_ref(inter, "Vino")
        _data_restriction(inter, OWL_QUALITY_PROP, "integer", lo, True, hi, True)
        qnames.append(name)

    # --- assiomi di disgiunzione ----------------------------------------------------
    groups = list(disjoint_groups().values()) + [qnames]
    for names in groups:
        dj = _sub(root, OWL, "AllDisjointClasses")
        members = _sub(dj, OWL, "members", {q(RDF, "parseType"): "Collection"})
        for n in names:
            _class_ref(members, n)

    return ET.ElementTree(root)


def summarize(path: Path = cfg.ONTOLOGY_PATH) -> dict:
    root = ET.parse(path).getroot()
    classes = [e for e in root.findall(q(OWL, "Class")) if e.get(q(RDF, "about"))]
    return {
        "classi": len(classes),
        "classi_definite": sum(1 for e in classes if e.find(q(OWL, "equivalentClass")) is not None),
        "data_property": len(root.findall(q(OWL, "DatatypeProperty"))),
        "object_property": len(root.findall(q(OWL, "ObjectProperty"))),
        "assiomi_disgiunzione": len(root.findall(q(OWL, "AllDisjointClasses"))),
    }


def build(path: Path = cfg.ONTOLOGY_PATH) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_tree()
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return summarize(path)


def inspect(path: Path = cfg.ONTOLOGY_PATH) -> None:
    info = summarize(path)
    print(f"{path}")
    for k, v in info.items():
        print(f"  {k:22s} {v}")
    try:
        from owlready2 import get_ontology
    except ImportError:
        print("owlready2 non installato: ispezione con owlready2 saltata (pip install owlready2)")
        return
    onto = get_ontology(Path(path).resolve().as_uri()).load()
    print(f"  [owlready2] classi = {len(list(onto.classes()))}, "
          f"data property = {len(list(onto.data_properties()))}")


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        inspect()
    else:
        info = build()
        print(f"Ontologia scritta in {cfg.ONTOLOGY_PATH}")
        print("  " + ", ".join(f"{k}={v}" for k, v in info.items()))
