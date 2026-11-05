"""
Configurazione globale del progetto ENOKON.

Tutti i parametri sperimentali (seed, numero di run, iperparametri, frazione di
etichette per il semi-supervisionato, ...) sono qui: per ripetere gli esperimenti
con altre impostazioni basta cambiare questo file (o usare le opzioni di main.py).
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Percorsi
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "wine_quality.csv"          # generato da data/download_dataset.py
ONTOLOGY_PATH = ROOT / "ontology" / "wine.owl"     # generato da ontology/build_ontology.py
KB_PATH = ROOT / "kb" / "wine_kb.pl"               # generato da kb/build_kb.py
RESULTS_DIR = ROOT / "results"
REPORT_PATH = ROOT / "docs" / "relazione.md"

# IRI dell'ontologia (i nomi delle classi sono IRI + "#" + nome)
ONTOLOGY_IRI = "http://example.org/enokon/wine"

# ----------------------------------------------------------------------------
# Dataset (UCI Wine Quality: vinho verde rosso + bianco)
# ----------------------------------------------------------------------------
RAW_FEATURES = [
    "fixed_acidity",         # g/L  (come acido tartarico)
    "volatile_acidity",      # g/L  (come acido acetico)
    "citric_acid",           # g/L
    "residual_sugar",        # g/L
    "chlorides",             # g/L  (come NaCl)
    "free_sulfur_dioxide",   # mg/L
    "total_sulfur_dioxide",  # mg/L
    "density",               # g/cm3
    "pH",
    "sulphates",             # g/L  (come K2SO4)
    "alcohol",               # % vol
]
TYPE_COLUMN = "wine_type"      # nel CSV: 'red' | 'white'
TIPO_COLUMN = "tipo"           # derivata: 'rosso' | 'bianco' (usata da ontologia e KB)
QUALITY_COLUMN = "quality"     # punteggio sensoriale 0-10 (nel dataset: 3-9)

# Discretizzazione del punteggio di qualità nelle classi bersaglio
CLASS_NAMES = ["bassa", "media", "alta"]
LOW_MAX = 5                    # qualità <= 5  -> "bassa"
HIGH_MIN = 7                   # qualità >= 7  -> "alta"   (6 -> "media")

# Il dataset UCI contiene molti record identici (documentato in letteratura):
# con split casuali finirebbero sia in train sia in test gonfiando le metriche.
DROP_DUPLICATES = True

# ----------------------------------------------------------------------------
# Protocollo sperimentale
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
N_RUNS = 10                    # ripetizioni con split stratificati diversi
TEST_SIZE = 0.30
LABELED_FRACTION = 0.20        # quota di training con etichetta (BNB e BNB-EM)
CURVE_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]   # curva di apprendimento
CURVE_LAMBDAS = [1.0, 0.1]     # pesi dei non etichettati confrontati nella curva (BNB-EM)
NEMENYI_ALPHA = 0.05

# ----------------------------------------------------------------------------
# Iperparametri
# ----------------------------------------------------------------------------
DT_PARAMS = dict(max_depth=8, min_samples_leaf=5, class_weight="balanced")
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                 class_weight="balanced_subsample", n_jobs=-1)
LR_PARAMS = dict(C=1.0, max_iter=3000, class_weight="balanced")

# Griglie per la ricerca opzionale (main.py --tune): CV interna a 3 fold sul solo training
TUNE_GRIDS = {
    "DT": {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [1, 5, 20]},
    "RF": {"min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.5]},
    "LR": {"clf__C": [0.1, 1.0, 10.0]},
}

BNB_ALPHA = 1.0                # smoothing di Laplace (Bernoulli NB / BNB-EM)
EM_LAMBDA = 1.0                # peso dei dati non etichettati nell'EM (Nigam et al., 2000)
EM_MAX_ITER = 100
EM_TOL = 1e-5                  # tolleranza relativa sulla log-verosimiglianza
TAN_ALPHA = 1.0                # smoothing di Laplace nelle CPT della rete bayesiana
