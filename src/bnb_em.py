"""
Bernoulli Naive Bayes semi-supervisionato con EM (Nigam, McCallum, Thrun, Mitchell, 2000).

Si dispone di un piccolo insieme etichettato L e di un grande insieme non etichettato U.

  inizializzazione : M-step sui soli esempi di L
  E-step           : per x in U,  gamma_c(x) = P(c | x, theta)  (responsabilità)
  M-step           : conteggi soft  N_c   = sum_{L} 1[y=c]   + lambda * sum_{U} gamma_c
                                    N_cj  = sum_{L} 1[y=c] x_j + lambda * sum_{U} gamma_c x_j
  convergenza      : variazione relativa della log-verosimiglianza < tol

  log L = sum_{L} log P(x, y)  +  lambda * sum_{U} log sum_c P(x, c)

lambda in [0, 1] pesa il contributo dei non etichettati (lambda = 1: peso pieno).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from src.bernoulli_nb import BernoulliNB


class SemiSupervisedBernoulliNB(BernoulliNB):
    def __init__(self, n_classes: int, alpha: float = 1.0, lambda_u: float = 1.0,
                 max_iter: int = 100, tol: float = 1e-5):
        super().__init__(n_classes, alpha)
        self.lambda_u = float(lambda_u)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_iter_ = 0
        self.log_likelihoods_: list[float] = []

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray | None = None
            ) -> "SemiSupervisedBernoulliNB":
        X_l = np.asarray(X_l)
        R_l = np.zeros((len(y_l), self.n_classes))
        R_l[np.arange(len(y_l)), y_l] = 1.0

        # 1) inizializzazione supervisionata
        self._m_step(X_l, R_l)
        self.log_likelihoods_ = []
        self.n_iter_ = 0
        if X_u is None or len(X_u) == 0 or self.lambda_u == 0.0:
            return self

        X_u = np.asarray(X_u)
        X_all = np.vstack([X_l, X_u]).astype(float)
        w = np.concatenate([np.ones(len(X_l)), np.full(len(X_u), self.lambda_u)])
        prev = -np.inf
        for it in range(1, self.max_iter + 1):
            # 2) E-step sui non etichettati
            lj_u = self.log_joint(X_u)
            lse = logsumexp(lj_u, axis=1)
            R_u = np.exp(lj_u - lse[:, None])
            ll = float(self.log_joint(X_l)[np.arange(len(y_l)), y_l].sum()
                       + self.lambda_u * lse.sum())
            self.log_likelihoods_.append(ll)
            # 3) M-step con conteggi soft
            self._m_step(X_all, np.vstack([R_l, R_u]), w)
            self.n_iter_ = it
            if np.isfinite(prev) and abs(ll - prev) <= self.tol * abs(ll):
                break
            prev = ll
        return self
