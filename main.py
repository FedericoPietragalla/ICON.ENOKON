#!/usr/bin/env python3
"""
ENOKON - classificazione della qualità del vino con ML + conoscenza di dominio.

Pipeline completa:

  1. rigenera ontologia OWL e KB Prolog dalla conoscenza in src/knowledge.py
  2. carica il dataset (lo scarica da UCI se manca) e calcola le feature OWL
  3. esperimento principale: N run con split stratificati 70/30
        DT / RF / LR  su feature Raw e Raw+OWL
        BNB e BNB-EM (semi-supervisionati) e rete bayesiana TAN su feature OWL
        KB Prolog (regole di esperto, nessun apprendimento)
  4. test di Friedman + post-hoc di Nemenyi sulla F1 macro
  5. curva di apprendimento semi-supervisionato (BNB vs BNB-EM al variare delle etichette)
  6. salvataggio di tabelle (CSV, summary.md) e grafici (PNG) in results/

Uso:
    python main.py                       # esperimento completo (10 run)
    python main.py --runs 3 --no-curve   # prova veloce
    python main.py --tune                # ricerca degli iperparametri (più lento)
    python main.py --update-report       # scrive tabelle e risultati in docs/relazione.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from kb import build_kb  # noqa: E402
from ontology import build_ontology  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import preprocessing as prep  # noqa: E402
from src import visualization as viz  # noqa: E402
from src.bayesian_network import TreeAugmentedNaiveBayes  # noqa: E402
from src.bernoulli_nb import BernoulliNB  # noqa: E402
from src.bnb_em import SemiSupervisedBernoulliNB  # noqa: E402
from src.ontology_features import (build_concepts, catalog_markdown, concept_catalog,  # noqa: E402
                                   coverage_by_class, owl_feature_matrix, owl_feature_names)
from src.prolog_kb import WineKB  # noqa: E402
from src.supervised import SUPERVISED_MODELS, build_estimator  # noqa: E402

K = len(cfg.CLASS_NAMES)
MODEL_ORDER = ["DT", "RF", "LR", "DT+OWL", "RF+OWL", "LR+OWL", "BNB", "BNB-EM", "TAN", "KB"]


def model_info(labeled_fraction: float) -> dict:
    lf = f"{labeled_fraction:.0%}"
    return {
        "DT": ("Raw", "100%"), "RF": ("Raw", "100%"), "LR": ("Raw", "100%"),
        "DT+OWL": ("Raw+OWL", "100%"), "RF+OWL": ("Raw+OWL", "100%"), "LR+OWL": ("Raw+OWL", "100%"),
        "BNB": ("OWL", lf), "BNB-EM": ("OWL", f"{lf} + non etichettati"),
        "TAN": ("OWL", "100%"), "KB": ("OWL + regole", "nessuna (non appresa)"),
    }


def log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================ esperimento
def run_experiment(y, X_raw, X_owl, kb_pred, args):
    X_raw_owl = np.hstack([X_raw, X_owl.astype(float)])
    splits = prep.make_splits(y, args.runs, cfg.TEST_SIZE, args.seed)
    records = []
    cms = {m: np.zeros((K, K), dtype=int) for m in MODEL_ORDER}
    em_iters, tan_edges = [], None

    for run, (tr, te) in enumerate(splits):
        t0 = time.time()
        seed_r = args.seed + run
        preds = {}

        # --- modelli supervisionati classici, su due insiemi di feature
        for X, suffix in ((X_raw, ""), (X_raw_owl, "+OWL")):
            for m in SUPERVISED_MODELS:
                est = build_estimator(m, seed_r, tune=args.tune)
                est.fit(X[tr], y[tr])
                preds[m + suffix] = est.predict(X[te])

        # --- Bernoulli NB e BNB-EM: solo una frazione del training ha l'etichetta
        lab = prep.labeled_mask(y[tr], args.labeled_fraction, seed_r)
        Xtr, ytr = X_owl[tr], y[tr]
        bnb = BernoulliNB(K, cfg.BNB_ALPHA).fit(Xtr[lab], ytr[lab])
        preds["BNB"] = bnb.predict(X_owl[te])
        em = SemiSupervisedBernoulliNB(K, cfg.BNB_ALPHA, cfg.EM_LAMBDA, cfg.EM_MAX_ITER, cfg.EM_TOL)
        em.fit(Xtr[lab], ytr[lab], Xtr[~lab])
        preds["BNB-EM"] = em.predict(X_owl[te])
        em_iters.append(em.n_iter_)

        # --- rete bayesiana TAN (tutte le etichette del training)
        tan = TreeAugmentedNaiveBayes(K, cfg.TAN_ALPHA, owl_feature_names()).fit(Xtr, ytr)
        preds["TAN"] = tan.predict(X_owl[te])
        if run == 0:
            tan_edges = tan.edges()

        # --- KB Prolog: nessun addestramento, si valutano le stesse righe di test
        preds["KB"] = kb_pred[te]

        for name, p in preds.items():
            rec = ev.compute_metrics(y[te], p, K)
            rec.update(run=run, model=name)
            records.append(rec)
            cms[name] += ev.confusion(y[te], p, K)
        f1s = {m: next(r["f1"] for r in records if r["run"] == run and r["model"] == m) for m in ("RF", "BNB", "BNB-EM", "TAN", "KB")}
        log(f"  run {run + 1:>2}/{len(splits)}  ({time.time() - t0:4.1f}s)  F1: "
            + "  ".join(f"{k}={v:.3f}" for k, v in f1s.items()))
    return pd.DataFrame(records), cms, tan_edges, em_iters


def em_name(lam: float) -> str:
    return "BNB-EM" if lam == cfg.EM_LAMBDA else f"BNB-EM (λ={lam:g})"


def run_curve(y, X_owl, args):
    """Curva di apprendimento: BNB (solo etichettati) vs BNB-EM (etichettati + non etichettati),
    con diversi pesi lambda dei dati non etichettati."""
    splits = prep.make_splits(y, args.runs, cfg.TEST_SIZE, args.seed)
    lambdas = list(dict.fromkeys([cfg.EM_LAMBDA] + list(cfg.CURVE_LAMBDAS)))
    rows = []
    for run, (tr, te) in enumerate(splits):
        seed_r = args.seed + run
        for frac in cfg.CURVE_FRACTIONS:
            lab = prep.labeled_mask(y[tr], frac, seed_r)
            Xl, yl, Xu = X_owl[tr][lab], y[tr][lab], X_owl[tr][~lab]
            models = {"BNB": BernoulliNB(K, cfg.BNB_ALPHA).fit(Xl, yl)}
            for lam in lambdas:
                models[em_name(lam)] = SemiSupervisedBernoulliNB(
                    K, cfg.BNB_ALPHA, lam, cfg.EM_MAX_ITER, cfg.EM_TOL).fit(Xl, yl, Xu)
            for name, mdl in models.items():
                m = ev.compute_metrics(y[te], mdl.predict(X_owl[te]), K)
                rows.append({"run": run, "fraction": frac, "model": name, "f1": m["f1"],
                             "gmean": m["gmean"], "accuracy": m["accuracy"]})
    return pd.DataFrame(rows)


# ============================================================================== report
def ms(mean, std) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def build_summary(info, ds, summary, stats_res, curve, kb, kb_rules, cms, args) -> str:
    L = []
    w = L.append
    w("### Dataset ed esperimento")
    w("")
    w(f"- Vini dopo la pulizia: **{ds['n']}** ({ds['n_red']} rossi, {ds['n_white']} bianchi); "
      f"rimossi {ds['n_duplicates']} duplicati su {ds['n_loaded']} record.")
    cc = ds["class_counts"]
    w("- Classi: " + ", ".join(f"{k} = {v} ({100 * v / ds['n']:.1f}%)" for k, v in cc.items()) + ".")
    w(f"- Protocollo: {args.runs} run con split stratificato {int((1 - cfg.TEST_SIZE) * 100)}/{int(cfg.TEST_SIZE * 100)} "
      f"(seed base {args.seed}); BNB/BNB-EM con {args.labeled_fraction:.0%} di esempi etichettati; "
      f"iperparametri {'ottimizzati con CV interna' if args.tune else 'fissi (config.py)'}.")
    w("")
    w("### Risultati (media ± deviazione standard sui run)")
    w("")
    w("| Modello | Feature | Etichette | Precision | Recall | F1 macro | G-Mean |")
    w("|---|---|---|---|---|---|---|")
    for m in summary.index:
        fs, lab = info[m]
        row = [ms(summary.loc[m, ("mean", k)], summary.loc[m, ("std", k)]) for k in ("precision", "recall", "f1", "gmean")]
        w(f"| {m} | {fs} | {lab} | " + " | ".join(row) + " |")
    w("")
    w("### Confronto statistico (Friedman + Nemenyi sulla F1 macro)")
    w("")
    if stats_res["mean_ranks"] is None:
        w(f"Test non eseguito: servono almeno 3 run e 3 modelli (run = {stats_res['n_runs']}).")
    else:
        sig = "**significative**" if stats_res["pvalue"] < stats_res["alpha"] else "non significative"
        w(f"Friedman: χ² = {stats_res['statistic']:.2f}, p = {stats_res['pvalue']:.2e} → differenze {sig} "
          f"(α = {stats_res['alpha']}). Differenza critica di Nemenyi CD = {stats_res['cd']:.2f} ranghi.")
        w("")
        w("| Modello | Rango medio (1 = migliore) |")
        w("|---|---|")
        for m, r in stats_res["mean_ranks"].items():
            w(f"| {m} | {r:.2f} |")
        w("")
        pv = stats_res["nemenyi"]
        pairs = [(a, b, pv.loc[a, b]) for i, a in enumerate(pv.index) for b in pv.columns[i + 1:] if pv.loc[a, b] < stats_res["alpha"]]
        if pairs:
            w("Coppie di modelli con differenza significativa (Nemenyi): "
              + "; ".join(f"{a} vs {b} (p = {p:.3f})" for a, b, p in sorted(pairs, key=lambda t: t[2])) + ".")
        else:
            w("Nessuna coppia di modelli risulta significativamente diversa al post-hoc di Nemenyi.")
    w("")
    if curve is not None:
        w("### Apprendimento semi-supervisionato: BNB vs BNB-EM (F1 macro)")
        w("")
        cmodels = list(dict.fromkeys(curve["model"]))
        w("| % etichettati | " + " | ".join(cmodels) + " |")
        w("|---|" + "---|" * len(cmodels))
        g = curve.groupby(["fraction", "model"])["f1"].agg(["mean", "std"]).fillna(0.0)
        for frac in sorted(curve["fraction"].unique()):
            w(f"| {frac:.0%} | " + " | ".join(ms(g.loc[(frac, m), "mean"], g.loc[(frac, m), "std"]) for m in cmodels) + " |")
        w("")
        w("λ è il peso dei dati non etichettati nell'EM (λ = 1: peso pieno; λ piccolo: l'EM li considera meno).")
        w("")
    w("### Knowledge Base: regole di classificazione")
    w("")
    engine = kb.engine_used + (f", accordo con il motore Python {kb.agreement:.1%}" if kb.agreement is not None else "")
    w(f"Motore di esecuzione della KB: **{engine}**."
      + (f" (motivo: {kb.warning})" if kb.engine_used == "python" and kb.warning else ""))
    w("")
    w("| Regola | Classe | Vini decisi | Copertura % | Precisione % |")
    w("|---|---|---|---|---|")
    for _, r in kb_rules.iterrows():
        prec = "-" if pd.isna(r["precisione_%"]) else f"{r['precisione_%']:.1f}"
        w(f"| {r['regola']} | {r['classe']} | {r['vini_decisi']} | {r['copertura_%']:.1f} | {prec} |")
    w("")
    return "\n".join(L)


def replace_between(text: str, start: str, end: str, new: str):
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        return text, False
    return text[: i + len(start)] + "\n" + new.strip() + "\n" + text[j:], True


def update_report(summary_md: str, concepts_md: str) -> None:
    path = cfg.REPORT_PATH
    if not path.exists():
        log(f"  (relazione non trovata: {path})")
        return
    text = path.read_text(encoding="utf-8")
    text, ok1 = replace_between(text, "<!-- RISULTATI:INIZIO -->", "<!-- RISULTATI:FINE -->", summary_md)
    text, ok2 = replace_between(text, "<!-- CONCETTI:INIZIO -->", "<!-- CONCETTI:FINE -->", concepts_md)
    path.write_text(text, encoding="utf-8")
    log(f"  relazione aggiornata: {path}  (risultati: {'ok' if ok1 else 'marcatori mancanti'}, "
        f"concetti: {'ok' if ok2 else 'marcatori mancanti'})")


# ================================================================================ main
def parse_args():
    ap = argparse.ArgumentParser(description="ENOKON - pipeline completa", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--data", default=str(cfg.DATA_PATH), help="CSV del dataset (default: data/wine_quality.csv)")
    ap.add_argument("--output-dir", default=str(cfg.RESULTS_DIR), help="cartella dei risultati (default: results/)")
    ap.add_argument("--runs", type=int, default=cfg.N_RUNS, help=f"numero di run (default {cfg.N_RUNS})")
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    ap.add_argument("--labeled-fraction", type=float, default=cfg.LABELED_FRACTION,
                    help="quota di training etichettata per BNB/BNB-EM")
    ap.add_argument("--tune", action="store_true", help="ottimizza gli iperparametri di DT/RF/LR (CV interna)")
    ap.add_argument("--no-curve", action="store_true", help="salta la curva di apprendimento semi-supervisionato")
    ap.add_argument("--kb-engine", choices=["auto", "swipl", "python"], default="auto")
    ap.add_argument("--update-report", action="store_true", help="aggiorna le sezioni automatiche di docs/relazione.md")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("[1/7] Conoscenza di dominio: rigenero ontologia OWL e KB Prolog da src/knowledge.py")
    try:
        o = build_ontology.build()
        build_kb.build()
        log(f"      wine.owl: {o['classi']} classi ({o['classi_definite']} definite), "
            f"{o['data_property']} data property, {o['assiomi_disgiunzione']} assiomi di disgiunzione")
    except OSError as exc:
        log(f"      (impossibile riscrivere i file: {exc}; uso quelli esistenti)")

    log("[2/7] Dataset e feature OWL")
    data_path = Path(args.data)
    if not data_path.exists() and data_path.resolve() == cfg.DATA_PATH.resolve():
        log("      dataset assente: lo scarico da UCI...")
        from data import download_dataset
        download_dataset.build(cfg.DATA_PATH)
    df = prep.load_dataset(data_path)
    ds = prep.dataset_summary(df)
    y = df["quality_class"].to_numpy()
    concepts = build_concepts(df)
    X_owl = owl_feature_matrix(concepts)
    X_raw = prep.raw_feature_matrix(df)
    log(f"      {ds['n']} vini ({ds['n_red']} rossi / {ds['n_white']} bianchi), duplicati rimossi: {ds['n_duplicates']}")
    log("      classi: " + ", ".join(f"{k}={v}" for k, v in ds["class_counts"].items()))
    log(f"      feature: {X_raw.shape[1]} raw + {X_owl.shape[1]} concetti OWL")

    cov = coverage_by_class(concepts, y)
    cov.to_csv(out / "owl_concept_coverage.csv", float_format="%.2f", encoding="utf-8")
    concept_catalog(concepts).to_csv(out / "concept_catalog.csv", index=False, encoding="utf-8")
    viz.plot_class_distribution(df, fig_dir / "class_distribution.png")
    viz.plot_correlation(df, fig_dir / "correlation.png")
    viz.plot_owl_coverage(cov, fig_dir / "owl_coverage.png")

    log("[3/7] Knowledge Base Prolog")
    kb = WineKB(cfg.KB_PATH, engine=args.kb_engine)
    try:
        kb_pred = kb.predict(df, concepts)
    except RuntimeError as exc:
        sys.exit(f"\nErrore: {exc}\nUsa --kb-engine auto (o python) per ripiegare sul motore Python.")
    kb_rules = kb.rule_report(y)
    kb_rules.to_csv(out / "kb_rules.csv", index=False, encoding="utf-8")
    log(f"      motore: {kb.engine_used}"
        + (f" (accordo con Python: {kb.agreement:.2%})" if kb.agreement is not None else f"  [{kb.warning}]"))

    log(f"[4/7] Esperimento principale: {args.runs} run, {len(MODEL_ORDER)} modelli"
        + (", con ottimizzazione iperparametri" if args.tune else ""))
    runs_df, cms, tan_edges, em_iters = run_experiment(y, X_raw, X_owl, kb_pred, args)
    log(f"      iterazioni EM medie: {np.mean(em_iters):.1f}")

    log("[5/7] Aggregazione e test statistici")
    info = model_info(args.labeled_fraction)
    summary = ev.summarize(runs_df, MODEL_ORDER)
    flat = summary.copy()
    flat.columns = [f"{m}_{s}" for s, m in flat.columns]
    flat.insert(0, "etichette", [info[m][1] for m in flat.index])
    flat.insert(0, "feature", [info[m][0] for m in flat.index])
    flat.to_csv(out / "results.csv", float_format="%.4f", index_label="model", encoding="utf-8")
    runs_df.to_csv(out / "results_per_run.csv", index=False, float_format="%.4f", encoding="utf-8")

    f1_matrix = runs_df.pivot(index="run", columns="model", values="f1")[summary.index.tolist()]
    stats_res = ev.friedman_nemenyi(f1_matrix, cfg.NEMENYI_ALPHA)
    if stats_res["mean_ranks"] is not None:
        stats_res["nemenyi"].to_csv(out / "nemenyi_pvalues.csv", float_format="%.5f", encoding="utf-8")
        stats_res["mean_ranks"].to_csv(out / "mean_ranks.csv", header=["rango_medio"], index_label="model", encoding="utf-8")

    curve = None
    if not args.no_curve:
        log("[6/7] Curva di apprendimento semi-supervisionato")
        curve = run_curve(y, X_owl, args)
        curve.to_csv(out / "semi_supervised_curve.csv", index=False, float_format="%.4f", encoding="utf-8")
    else:
        log("[6/7] Curva di apprendimento: saltata (--no-curve)")

    log("[7/7] Grafici e riepilogo")
    viz.plot_metrics(summary, fig_dir / "metrics.png")
    viz.plot_f1_boxplot(runs_df, summary.index.tolist(), fig_dir / "f1_boxplot.png")
    if stats_res["nemenyi"] is not None:
        viz.plot_nemenyi(stats_res["nemenyi"], cfg.NEMENYI_ALPHA, fig_dir / "nemenyi.png")
    best = summary[("mean", "f1")].idxmax()
    for name in dict.fromkeys([best, "KB"]):
        viz.plot_confusion(cms[name], f"Matrice di confusione ({name}, somma sui run)",
                           fig_dir / f"confusion_{name.replace('+', '_plus_')}.png")
    viz.plot_tan_structure(tan_edges, owl_feature_names(), fig_dir / "tan_structure.png")
    if curve is not None:
        viz.plot_semi_supervised_curve(curve, fig_dir / "semi_supervised_curve.png")

    summary_md = build_summary(info, ds, summary, stats_res, curve, kb, kb_rules, cms, args)
    (out / "summary.md").write_text("## Risultati sperimentali ENOKON\n\n" + summary_md, encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'Modello':<9}{'Precision':>16}{'Recall':>16}{'F1 macro':>16}{'G-Mean':>16}")
    print("-" * 78)
    for m in summary.index:
        cells = [f"{summary.loc[m, ('mean', k)]:.3f}+/-{summary.loc[m, ('std', k)]:.3f}" for k in ("precision", "recall", "f1", "gmean")]
        print(f"{m:<9}" + "".join(f"{c:>16}" for c in cells))
    print("=" * 78)
    if stats_res["mean_ranks"] is not None:
        print(f"Friedman: chi2={stats_res['statistic']:.2f}  p={stats_res['pvalue']:.2e}   CD(Nemenyi)={stats_res['cd']:.2f}")
    print(f"Miglior modello per F1 macro: {best}")
    print(f"Risultati in: {out}   (tempo totale {time.time() - t_start:.0f}s)")

    if args.update_report:
        update_report(summary_md, catalog_markdown(concepts))


if __name__ == "__main__":
    main()
