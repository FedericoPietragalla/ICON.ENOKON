"""
Rete bayesiana discreta: Tree-Augmented Naive Bayes (TAN), Friedman, Geiger & Goldszmidt (1997).

Struttura: la classe C è genitore di ogni feature binaria X_j (come nel Naive Bayes) e
in più le feature formano un ALBERO: ogni feature ha al più un altro genitore feature.
Questo permette di modellare le dipendenze tra concetti correlati (ad esempio i concetti
mutuamente esclusivi di una stessa famiglia, che il Naive Bayes tratta come indipendenti).

Apprendimento
  1. informazione mutua condizionata  I(X_i ; X_j | C)  per ogni coppia di feature
  2. albero di copertura di peso massimo (algoritmo di Prim), orientato dalla radice
  3. tabelle di probabilità condizionata (CPT) con smoothing di Laplace:
        P(X_root | C)        P(X_j | C, X_parent(j))
Inferenza (esatta, grazie alla fattorizzazione)
  P(c | x)  ∝  P(c) * prod_j P(x_j | c, x_parent(j))
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


class TreeAugmentedNaiveBayes:
    def __init__(self, n_classes: int, alpha: float = 1.0,
                 feature_names: Optional[Sequence[str]] = None):
        self.n_classes = int(n_classes)
        self.alpha = float(alpha)
        self.feature_names = list(feature_names) if feature_names is not None else None

    # ------------------------------------------------------------ apprendimento
    def _conditional_mutual_information(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, d = X.shape
        a = self.alpha
        Xf = X.astype(float)
        cmi = np.zeros((d, d))
        for c in range(self.n_classes):
            Xc = Xf[y == c]
            nc = len(Xc)
            if nc == 0:
                continue
            n11 = Xc.T @ Xc                       # x_i = 1, x_j = 1
            n10 = Xc.T @ (1.0 - Xc)               # x_i = 1, x_j = 0
            n01 = n10.T                           # x_i = 0, x_j = 1
            n00 = nc - n11 - n10 - n01
            den = nc + 4.0 * a
            p11, p10, p01, p00 = ((n11 + a) / den, (n10 + a) / den,
                                  (n01 + a) / den, (n00 + a) / den)
            pi1, pj1 = p11 + p10, p11 + p01       # marginali dal congiunto smussato
            pi0, pj0 = 1.0 - pi1, 1.0 - pj1
            mi = (p11 * np.log(p11 / (pi1 * pj1)) + p10 * np.log(p10 / (pi1 * pj0))
                  + p01 * np.log(p01 / (pi0 * pj1)) + p00 * np.log(p00 / (pi0 * pj0)))
            cmi += (nc / n) * mi
        np.fill_diagonal(cmi, 0.0)
        return cmi

    @staticmethod
    def _maximum_spanning_tree(w: np.ndarray, root: int) -> np.ndarray:
        """Algoritmo di Prim (massimo): restituisce il vettore dei genitori (-1 = radice)."""
        d = len(w)
        parent = np.full(d, -1, dtype=int)
        in_tree = np.zeros(d, dtype=bool)
        in_tree[root] = True
        best = w[root].astype(float).copy()
        best_parent = np.full(d, root, dtype=int)
        for _ in range(d - 1):
            cand = np.where(in_tree, -np.inf, best)
            j = int(np.argmax(cand))
            in_tree[j] = True
            parent[j] = best_parent[j]
            upd = (~in_tree) & (w[j] > best)
            best[upd] = w[j][upd]
            best_parent[upd] = j
        return parent

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeAugmentedNaiveBayes":
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape
        K, a = self.n_classes, self.alpha

        self.cmi_ = self._conditional_mutual_information(X, y)
        root = int(np.argmax(self.cmi_.sum(axis=1)))       # feature più "centrale"
        self.root_ = root
        self.parent_ = self._maximum_spanning_tree(self.cmi_, root)

        counts = np.bincount(y, minlength=K).astype(float)
        self.class_log_prior_ = np.log((counts + a) / (n + K * a))

        # CPT: theta_[j][c, v] = P(X_j = 1 | C = c, X_parent(j) = v)   (la radice ignora v)
        self.theta_ = []
        for j in range(d):
            th = np.zeros((K, 2))
            p = self.parent_[j]
            for c in range(K):
                in_c = y == c
                if p < 0:
                    nm, n1 = in_c.sum(), X[in_c, j].sum()
                    th[c, :] = (n1 + a) / (nm + 2.0 * a)
                else:
                    for v in (0, 1):
                        m = in_c & (X[:, p] == v)
                        th[c, v] = (X[m, j].sum() + a) / (m.sum() + 2.0 * a)
            self.theta_.append(th)
        return self

    # ---------------------------------------------------------------- inferenza
    def log_joint(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n, d = X.shape
        lj = np.tile(self.class_log_prior_, (n, 1))
        for j in range(d):
            p = self.parent_[j]
            pv = X[:, p].astype(int) if p >= 0 else np.zeros(n, dtype=int)
            t = self.theta_[j][:, pv].T                      # (n, K)
            xj = X[:, j][:, None].astype(float)
            lj += xj * np.log(t) + (1.0 - xj) * np.log1p(-t)
        return lj

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lj = self.log_joint(X)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.log_joint(X), axis=1)

    # --------------------------------------------------------------- struttura
    def edges(self) -> List[Tuple[str, str]]:
        """Archi dell'albero tra feature, come coppie (genitore, figlio)."""
        names = self.feature_names or [f"x{j}" for j in range(len(self.parent_))]
        return [(names[p], names[j]) for j, p in enumerate(self.parent_) if p >= 0]
