# ENOKON — Wine Knowledge-Based Quality Classification

Progetto di **Ingegneria della Conoscenza** che classifica la **qualità sensoriale del vino**
(bassa / media / alta) a partire dalle sue misure chimiche, combinando:

- **Apprendimento supervisionato** classico (Decision Tree, Random Forest, Logistic Regression)
- **Apprendimento semi-supervisionato** (Bernoulli Naive Bayes e BNB-EM, con algoritmo EM)
- **Ontologia OWL 2** come conoscenza di background (34 concetti enologici, di cui 31 usati come feature binarie)
- **Knowledge Base Prolog** con regole di esperto e spiegazione della decisione
- **Rete bayesiana discreta** (Tree-Augmented Naive Bayes, TAN)
- **Confronto statistico** tra i modelli (test di Friedman + post-hoc di Nemenyi)

> Tema: *vino*. Dataset: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
> (vinho verde rosso e bianco, 11 misure chimiche, punteggio sensoriale).

## Struttura del progetto

```
ENOKON/
├── main.py                    # pipeline completa (esperimenti, test statistici, grafici)
├── config.py                  # parametri: seed, run, iperparametri, soglie del protocollo
├── requirements.txt
├── data/
│   ├── download_dataset.py    # scarica UCI Wine Quality -> data/wine_quality.csv
│   └── wine_quality.csv       # (creato dallo script)
├── ontology/
│   ├── build_ontology.py      # genera wine.owl da src/knowledge.py
│   └── wine.owl               # ontologia OWL 2 (RDF/XML): 52 classi, 13 data property
├── kb/
│   ├── build_kb.py            # genera wine_kb.pl da src/knowledge.py
│   └── wine_kb.pl             # Knowledge Base Prolog (concetti + regole + spiegazione)
├── src/
│   ├── knowledge.py           # UNICA fonte della conoscenza di dominio (concetti, regole)
│   ├── preprocessing.py       # caricamento, pulizia, classi, split stratificati
│   ├── ontology_features.py   # concetti OWL come feature binarie, copertura per classe
│   ├── supervised.py          # DT / RF / LR (scikit-learn)
│   ├── bernoulli_nb.py        # Bernoulli Naive Bayes (numpy)
│   ├── bnb_em.py              # BNB semi-supervisionato con EM
│   ├── bayesian_network.py    # rete bayesiana TAN (numpy)
│   ├── prolog_kb.py           # esecuzione della KB (SWI-Prolog o motore Python)
│   ├── evaluation.py          # metriche, Friedman, Nemenyi
│   └── visualization.py       # grafici
├── tests/
│   └── test_consistency.py    # OWL/Prolog coerenti con Python; correttezza dei modelli
├── docs/
│   └── relazione.md           # relazione del progetto
└── results/                   # tabelle (CSV, summary.md) e grafici (figures/)
```

## Idea chiave: una sola fonte per la conoscenza

Concetti e regole sono definiti **una volta sola** in `src/knowledge.py`. Da lì si derivano, senza
duplicare nulla a mano, le feature binarie dei modelli, l'ontologia `wine.owl` e la KB `wine_kb.pl`.
Per cambiare una soglia basta modificare quel file: `main.py` rigenera ontologia e KB a ogni
esecuzione, e `tests/test_consistency.py` verifica che OWL e Prolog dicano esattamente ciò che dice Python.

La conoscenza ha tre livelli:

1. **Concetti di soglia** — discretizzano una misura (`AlcolAlto`: alcol ≥ 12 % vol, `PHBasso`, `Secco`, ...).
2. **Concetti composti** — nozioni enologiche con AND/OR (`VinoDifettoso`, `VinoEquilibrato`, `RischioInstabilita`, ...).
3. **Regole di classificazione** — sistema esperto in Prolog: `classifica(Vino, Classe, Motivo)`.

## Modelli confrontati

| Modello | Feature | Etichette usate |
|---|---|---|
| DT, RF, LR | misure chimiche + tipo (*Raw*) | 100 % |
| DT+OWL, RF+OWL, LR+OWL | *Raw* + concetti OWL | 100 % |
| BNB | concetti OWL | 20 % (configurabile) |
| BNB-EM | concetti OWL | 20 % + dati non etichettati (EM) |
| TAN | concetti OWL | 100 % |
| KB | concetti OWL + regole Prolog | nessuna (non appresa) |

Protocollo: 10 run con split stratificato 70/30; metriche macro (Precision, Recall, F1) e
**G-Mean**; Friedman + Nemenyi sulla F1. Inoltre una **curva di apprendimento** confronta BNB e
BNB-EM al variare della quota di esempi etichettati (1 % … 100 %).

## Installazione

Richiede **Python 3.9+**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Facoltativo: [SWI-Prolog](https://www.swi-prolog.org) (con `swipl` nel PATH) per eseguire la KB
davvero in Prolog. Se manca, `main.py` usa automaticamente un motore Python equivalente e lo segnala.

## Esecuzione

```bash
python data/download_dataset.py      # una volta sola (main.py lo fa da solo se il CSV manca)
python main.py                       # esperimento completo (10 run + curva di apprendimento)
```

Opzioni utili:

```bash
python main.py --runs 3 --no-curve        # prova veloce
python main.py --tune                     # ottimizza gli iperparametri di DT/RF/LR (CV interna, più lento)
python main.py --labeled-fraction 0.1     # BNB/BNB-EM con il 10 % di etichette
python main.py --kb-engine swipl          # forza SWI-Prolog (errore se non disponibile)
python main.py --update-report            # scrive tabelle e risultati dentro docs/relazione.md
python data/download_dataset.py --raw-dir <cartella>   # se hai già scaricato i due CSV UCI
```

Rigenerare ontologia e KB a mano, oppure ispezionare l'ontologia:

```bash
python ontology/build_ontology.py [--inspect]
python kb/build_kb.py
```

Provare la KB in SWI-Prolog (l'elenco completo dei fatti dell'esempio è in fondo a `kb/wine_kb.pl`):

```prolog
?- [kb/wine_kb].
?- assertz(wine(v1)), assertz(tipo(v1, bianco)), assertz(grado_alcolico(v1, 12.5)), ... .
?- classifica(v1, Classe, Motivo).
Classe = alta, Motivo = maturo_e_pulito.
```

## Output (cartella `results/`)

| File | Contenuto |
|---|---|
| `summary.md` | tutte le tabelle in Markdown (risultati, Friedman/Nemenyi, curva, regole KB) |
| `results.csv`, `results_per_run.csv` | metriche aggregate e per singolo run |
| `mean_ranks.csv`, `nemenyi_pvalues.csv` | confronto statistico |
| `semi_supervised_curve.csv` | curva di apprendimento BNB vs BNB-EM |
| `owl_concept_coverage.csv`, `concept_catalog.csv` | copertura e catalogo dei concetti |
| `kb_rules.csv` | copertura e precisione di ogni regola della KB |
| `figures/*.png` | distribuzione delle classi, correlazioni, copertura OWL, confronto modelli, boxplot F1, Nemenyi, matrici di confusione, struttura della rete TAN, curva semi-supervisionata |

> **I risultati non sono inclusi nel repository**: si generano con `python main.py` sul dataset reale.

## Test

```bash
python tests/test_consistency.py     # oppure:  python -m pytest tests -q
```

Verificano che `wine.owl` e `wine_kb.pl` esprimano esattamente le definizioni di `src/knowledge.py`
(compresi i valori *esattamente sulle soglie*), che i concetti mutuamente esclusivi non si
sovrappongano e che i modelli probabilistici siano corretti (BNB identico a scikit-learn, obiettivo
EM monotono, TAN che collega le feature dipendenti, valori critici di Nemenyi come in Demšar 2006).

## Note metodologiche

- **Duplicati**: il dataset UCI contiene molti record identici; vengono rimossi (`DROP_DUPLICATES`)
  perché con split casuali finirebbero sia in train sia in test, gonfiando le metriche.
- **Nessun data leakage dalla conoscenza**: i concetti sono funzioni deterministiche delle misure,
  senza parametri stimati dai dati; le soglie sono fissate a priori, non ottimizzate sulle etichette.
- **Soglie**: sono di tre tipi (normativo/sensoriale, convenzione enologica, empirica); ognuna ha il
  proprio fondamento in `src/knowledge.py` e nella relazione. Prima di consegnare conviene
  **verificarle e citare le fonti puntuali**.
- Il dataset riguarda solo il *vinho verde* portoghese: i risultati non si generalizzano ad altri vini.

## Dataset e citazione

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009). *Modeling wine preferences by data
mining from physicochemical properties.* Decision Support Systems 47(4):547–553.
UCI Machine Learning Repository: <https://doi.org/10.24432/C56S3T>

## Autore

`<Nome Cognome>` — `<matricola>` — Corso di Ingegneria della Conoscenza, A.A. `<anno>`
