"""
Naive Bayes di Bernoulli implementato da zero (numpy).

Modello:   P(c, x) = P(c) * prod_j  theta_cj^x_j * (1 - theta_cj)^(1 - x_j)

I parametri si stimano con smoothing di Laplace (alpha). Lo stesso M-step, scritto
in forma "pesata" (responsabilità soft per ogni classe), viene riusato da
SemiSupervisedBernoulliNB (src/bnb_em.py) per l'algoritmo EM.
"""
from __future__ import annotations

import numpy as np


class BernoulliNB:
    def __init__(self, n_classes: int, alpha: float = 1.0):
        self.n_classes = int(n_classes)
        self.alpha = float(alpha)

    # ---------------------------------------------------------------- stima
    def _m_step(self, X: np.ndarray, R: np.ndarray, w: np.ndarray | None = None) -> None:
        """Massimizzazione con conteggi soft.

        X : (n, d) binaria      R : (n, K) responsabilità (one-hot per gli etichettati)
        w : (n,) peso di ogni esempio (1 per gli etichettati, lambda per gli altri)
        """
        a, K = self.alpha, self.n_classes
        Rw = R if w is None else R * w[:, None]
        Nk = Rw.sum(axis=0)                                  # massa di ogni classe
        Njk = Rw.T @ X.astype(float)                         # massa con x_j = 1 per classe
        self.class_log_prior_ = np.log((Nk + a) / (Nk.sum() + K * a))
        theta = (Njk + a) / (Nk[:, None] + 2.0 * a)
        self.theta_ = theta
        self.log_theta_ = np.log(theta)
        self.log_1m_theta_ = np.log1p(-theta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNB":
        R = np.zeros((len(y), self.n_classes))
        R[np.arange(len(y)), y] = 1.0
        self._m_step(np.asarray(X), R)
        return self

    # ------------------------------------------------------------ inferenza
    def log_joint(self, X: np.ndarray) -> np.ndarray:
        """log P(c) + log P(x | c) per ogni classe, matrice (n, K)."""
        X = np.asarray(X, dtype=float)
        return (X @ self.log_theta_.T + (1.0 - X) @ self.log_1m_theta_.T
                + self.class_log_prior_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lj = self.log_joint(X)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.log_joint(X), axis=1)
