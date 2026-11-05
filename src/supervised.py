"""
Modelli supervisionati classici (scikit-learn): Decision Tree, Random Forest, Logistic Regression.

Vengono addestrati su due insiemi di feature:
  * "Raw"      : 11 misure chimiche + indicatore del tipo di vino
  * "Raw+OWL"  : le stesse + i concetti di conoscenza di background come feature binarie

Gli iperparametri di default sono in config.py. Con tune=True ogni modello esegue una
ricerca a griglia con CV interna a 3 fold (scoring F1 macro) SOLO sul training set.
"""
from __future__ import annotations

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config as cfg

SUPERVISED_MODELS = ("DT", "RF", "LR")


def build_estimator(name: str, random_state: int, tune: bool = False):
    if name == "DT":
        est = DecisionTreeClassifier(random_state=random_state, **cfg.DT_PARAMS)
    elif name == "RF":
        est = RandomForestClassifier(random_state=random_state, **cfg.RF_PARAMS)
    elif name == "LR":
        # la regressione logistica richiede feature su scale confrontabili
        est = Pipeline([("scale", StandardScaler()),
                        ("clf", LogisticRegression(random_state=random_state, **cfg.LR_PARAMS))])
    else:
        raise ValueError(f"Modello sconosciuto: {name}")

    if tune:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        est = GridSearchCV(est, cfg.TUNE_GRIDS[name], scoring="f1_macro", cv=cv, n_jobs=1)
    return est
