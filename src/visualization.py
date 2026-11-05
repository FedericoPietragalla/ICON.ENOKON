"""
Grafici del progetto (solo matplotlib, backend non interattivo: salva su file PNG).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from src.knowledge import CONCEPT_BY_NAME, FAMILIES

CLASS_COLORS = ["#b04a4a", "#d9a441", "#4f8a5b"]      # bassa, media, alta
METRIC_COLORS = {"precision": "#6c8ebf", "recall": "#82b366", "f1": "#d79b00", "gmean": "#ae4132"}
DPI = 150


def _save(fig, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _heatmap(ax, data: np.ndarray, xlabels, ylabels, cmap, fmt="{:.1f}", vmin=None, vmax=None,
             bold_mask=None, fontsize=8):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    lo = np.nanmin(data) if vmin is None else vmin
    hi = np.nanmax(data) if vmax is None else vmax
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            dark = (v - lo) / (hi - lo + 1e-12)
            color = "white" if (dark > 0.6 and cmap not in ("Blues_r",)) or (cmap == "Blues_r" and dark < 0.4) else "black"
            weight = "bold" if (bold_mask is not None and bold_mask[i, j]) else "normal"
            ax.text(j, i, fmt.format(v), ha="center", va="center", color=color,
                    fontsize=fontsize, fontweight=weight)
    return im


# ----------------------------------------------------------------------------- EDA
def plot_class_distribution(df: pd.DataFrame, path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    q = df[cfg.QUALITY_COLUMN].value_counts().sort_index()
    cols = [CLASS_COLORS[int(0 if s <= cfg.LOW_MAX else 2 if s >= cfg.HIGH_MIN else 1)] for s in q.index]
    ax.bar(q.index.astype(str), q.values, color=cols)
    for x, v in zip(range(len(q)), q.values):
        ax.text(x, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Punteggio sensoriale")
    ax.set_ylabel("Numero di vini")
    ax.set_title("Distribuzione del punteggio di qualità")

    ax = axes[1]
    ct = pd.crosstab(df[cfg.TIPO_COLUMN], df["quality_class"]).reindex(columns=range(len(cfg.CLASS_NAMES)), fill_value=0)
    bottom = np.zeros(len(ct))
    for k, name in enumerate(cfg.CLASS_NAMES):
        ax.bar(ct.index, ct[k].values, bottom=bottom, color=CLASS_COLORS[k], label=name)
        bottom += ct[k].values
    ax.set_title("Classi di qualità per tipo di vino")
    ax.set_ylabel("Numero di vini")
    ax.legend(title="Classe")
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation(df: pd.DataFrame, path) -> str:
    cols = cfg.RAW_FEATURES + ["is_red", cfg.QUALITY_COLUMN]
    corr = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = _heatmap(ax, corr, cols, cols, "coolwarm", fmt="{:.2f}", vmin=-1, vmax=1, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlazione di Pearson tra misure chimiche, tipo e qualità")
    return _save(fig, path)


# ----------------------------------------------------------------------------- OWL
def plot_owl_coverage(cov: pd.DataFrame, path) -> str:
    data = cov.to_numpy(dtype=float)
    labels = list(cov.columns)
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(cov) + 1.8))
    im = _heatmap(ax, data, labels, list(cov.index), "YlOrRd", fmt="{:.0f}", vmin=0, vmax=100, fontsize=7)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    fig.colorbar(im, ax=ax, fraction=0.04, label="% di vini che soddisfano il concetto")
    ax.set_title("Copertura dei concetti di conoscenza di background per classe")
    return _save(fig, path)


# ------------------------------------------------------------------------ risultati
def plot_metrics(summary: pd.DataFrame, path, title="Confronto tra i modelli (media ± dev. std sui run)") -> str:
    models = list(summary.index)
    metrics = ["precision", "recall", "f1", "gmean"]
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1 macro", "gmean": "G-Mean"}
    x = np.arange(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(models)), 4.8))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, summary[("mean", m)].to_numpy(), w, yerr=summary[("std", m)].to_numpy(),
               capsize=2, label=labels[m], color=METRIC_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valore")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return _save(fig, path)


def plot_f1_boxplot(runs: pd.DataFrame, order: Sequence[str], path) -> str:
    data = [runs.loc[runs["model"] == m, "f1"].to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(order)), 4.5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("F1 macro")
    ax.set_title("Distribuzione della F1 macro sui run")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_nemenyi(pvals: pd.DataFrame, alpha: float, path) -> str:
    data = pvals.to_numpy(dtype=float)
    k = len(pvals)
    fig, ax = plt.subplots(figsize=(0.75 * k + 3, 0.75 * k + 2))
    im = _heatmap(ax, data, list(pvals.columns), list(pvals.index), "Blues_r", fmt="{:.2f}",
                  vmin=0, vmax=1, bold_mask=data < alpha, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="p-value")
    ax.set_title(f"Post-hoc di Nemenyi (in grassetto: p < {alpha})")
    return _save(fig, path)


def plot_confusion(cm: np.ndarray, title: str, path) -> str:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = cfg.CLASS_NAMES
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.55 else "black", fontsize=9)
    ax.set_xlabel("Classe predetta")
    ax.set_ylabel("Classe reale")
    ax.set_title(title)
    return _save(fig, path)


def plot_tan_structure(edges: List[Tuple[str, str]], feature_names: Sequence[str], path) -> str:
    """Struttura della rete TAN: classe al centro (genitore di tutti) + albero tra i concetti."""
    n = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pos = {f: (np.cos(a), np.sin(a)) for f, a in zip(feature_names, angles)}
    palette = plt.get_cmap("tab20")
    fam_color = {fam: palette(i % 20) for i, fam in enumerate(FAMILIES)}

    fig, ax = plt.subplots(figsize=(11, 11))
    for f in feature_names:                                    # archi C -> X_j (Naive Bayes)
        ax.plot([0, pos[f][0]], [0, pos[f][1]], color="#999999", alpha=0.10, lw=0.8, zorder=1)
    for p, c in edges:                                         # archi dell'albero X_p -> X_c
        ax.annotate("", xy=pos[c], xytext=pos[p], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.3, shrinkA=9, shrinkB=9,
                                    connectionstyle="arc3,rad=0.18"))
    for f, a in zip(feature_names, angles):
        fam = CONCEPT_BY_NAME[f].family if f in CONCEPT_BY_NAME else ""
        ax.scatter(*pos[f], s=170, color=fam_color.get(fam, "#cccccc"), edgecolor="black", zorder=4)
        deg = np.degrees(a) % 360
        flip = 90 < deg < 270
        ax.text(1.09 * pos[f][0], 1.09 * pos[f][1], f, fontsize=8, ha="right" if flip else "left",
                va="center", rotation=deg + 180 if flip else deg, rotation_mode="anchor")
    ax.scatter(0, 0, s=900, color="#8c2d3f", edgecolor="black", zorder=5)
    ax.text(0, 0, "Qualità", color="white", ha="center", va="center", fontsize=9, zorder=6, fontweight="bold")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rete bayesiana TAN: la classe è genitore di ogni concetto (linee grigie);\n"
                 "le frecce nere sono le dipendenze apprese tra concetti (colore = famiglia)")
    return _save(fig, path)


def plot_semi_supervised_curve(curve: pd.DataFrame, path) -> str:
    """curve: colonne fraction, model, f1 (una riga per run e per frazione)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    palette = ["#6c8ebf", "#ae4132", "#d79b00", "#4f8a5b", "#7a5195"]
    markers = ["o", "s", "^", "D", "v"]
    for i, model in enumerate(dict.fromkeys(curve["model"])):
        g = curve[curve["model"] == model].groupby("fraction")["f1"]
        m, s = g.mean(), g.std(ddof=1).fillna(0.0)
        ax.plot(m.index * 100, m.values, marker=markers[i % 5], color=palette[i % 5], label=model)
        ax.fill_between(m.index * 100, m.values - s.values, m.values + s.values, color=palette[i % 5], alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("% di esempi di training con etichetta (scala logaritmica)")
    ax.set_ylabel("F1 macro sul test")
    ax.set_title("Apprendimento semi-supervisionato: BNB vs BNB-EM")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)
