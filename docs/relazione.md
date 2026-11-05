# ENOKON — Classificazione della qualità del vino con apprendimento automatico e conoscenza di dominio

**Autore:** `<Nome Cognome>` (`<matricola>`) · **Corso:** Ingegneria della Conoscenza · **A.A.** `<anno>`

> Le sezioni racchiuse tra i commenti `<!-- ...:INIZIO -->` e `<!-- ...:FINE -->` (tabella dei concetti e
> risultati) vengono riscritte da `python main.py --update-report`. Tutto il resto è testo da rivedere e
> completare, in particolare la sezione 9 (Discussione), che dipende dai risultati ottenuti.

---

## 1. Introduzione

Il progetto affronta un problema di classificazione: stabilire se un vino ha qualità **bassa**,
**media** o **alta** a partire da 11 misure chimiche e dal tipo (rosso/bianco). Il giudizio di
riferimento è sensoriale, quindi soggettivo e rumoroso: è uno scenario in cui la conoscenza di
dominio può affiancare l'apprendimento automatico.

Domande di ricerca:

- **RQ1** — I concetti enologici dell'ontologia, usati come feature aggiuntive, migliorano i classificatori classici?
- **RQ2** — Un classificatore generativo (Bernoulli Naive Bayes) trae vantaggio dai dati non etichettati tramite EM?
- **RQ3** — La rete bayesiana TAN supera il Naive Bayes modellando le dipendenze tra concetti?
- **RQ4** — Quanto si avvicina ai modelli appresi un sistema esperto fatto solo di regole (KB Prolog), che non vede mai le etichette?

Architettura complessiva:

```
                 misure chimiche + tipo  (CSV)
                            │
                  preprocessing (duplicati, classi)
                            │
            ┌───────────────┴────────────────┐
            │                                │
     feature Raw (12)               src/knowledge.py   <- unica fonte della conoscenza
            │                      ┌─────────┼───────────┐
            │                      ▼         ▼           ▼
            │                 wine.owl   wine_kb.pl   concetti binari (31)
            │                (ontologia)   (Prolog)          │
            └──────────┬───────────────────────┬────────────┘
                       ▼                       ▼
         DT / RF / LR (Raw, Raw+OWL)    BNB, BNB-EM, TAN, KB
                       └───────────┬───────────┘
                                   ▼
               N run → metriche → Friedman + Nemenyi → results/
```

## 2. Dominio e dataset

Il dataset è *Wine Quality* dell'UCI (Cortez et al., 2009): vini portoghesi *vinho verde*, **1599 rossi e
4898 bianchi**. Ogni vino è descritto da 11 misure fisico-chimiche e da un punteggio di qualità, che è la
mediana di almeno tre valutazioni sensoriali di esperti su una scala 0–10 (nel dataset i valori vanno da 3 a 9).

| Feature | Unità | Significato |
|---|---|---|
| `fixed_acidity` | g/L (acido tartarico) | acidi non volatili |
| `volatile_acidity` | g/L (acido acetico) | in eccesso dà note di aceto |
| `citric_acid` | g/L | contributo di freschezza |
| `residual_sugar` | g/L | zuccheri non fermentati |
| `chlorides` | g/L (NaCl) | contenuto salino |
| `free_sulfur_dioxide` | mg/L | SO2 libera: protezione antiossidante e antimicrobica |
| `total_sulfur_dioxide` | mg/L | SO2 libera + legata |
| `density` | g/cm³ | dipende da alcol e zuccheri |
| `pH` | — | acidità (scala logaritmica) |
| `sulphates` | g/L (K₂SO₄) | additivo con azione antimicrobica e antiossidante |
| `alcohol` | % vol | grado alcolico |

**Preparazione dei dati.** I due file (rossi, bianchi) sono uniti aggiungendo il tipo di vino. Il dataset
contiene molti **record duplicati**: con split casuali lo stesso vino finirebbe sia in training sia in test
e le metriche risulterebbero gonfiate, quindi i duplicati sono rimossi (il numero esatto è riportato in
`results/summary.md`). Il punteggio è discretizzato in tre classi ordinali: **bassa** (≤ 5), **media** (6),
**alta** (≥ 7). I punteggi estremi (3, 4, 8, 9) sono rari e darebbero classi quasi vuote. Le classi non sono
equilibrate (la distribuzione esatta è in `results/summary.md`): per questo si usano metriche macro, la
G-Mean e, dove disponibile, `class_weight="balanced"`.

**Limiti del dataset.** Il punteggio è soggettivo; i vini sono solo *vinho verde* portoghesi, quindi i
risultati non si generalizzano ad altre tipologie; rossi e bianchi hanno distribuzioni molto diverse.

## 3. Struttura del sistema

| Modulo | Ruolo |
|---|---|
| `src/knowledge.py` | **unica fonte** della conoscenza: concetti di soglia, concetti composti, regole di classificazione |
| `ontology/build_ontology.py` | genera `wine.owl` (OWL 2, RDF/XML) dalla conoscenza |
| `kb/build_kb.py` | genera `wine_kb.pl` (Prolog) dalla conoscenza |
| `src/ontology_features.py` | concetti → feature binarie, copertura per classe, catalogo |
| `src/preprocessing.py` | caricamento, pulizia, classi, split stratificati, sottoinsieme etichettato |
| `src/supervised.py` | DT, RF, LR (scikit-learn) |
| `src/bernoulli_nb.py`, `src/bnb_em.py` | Bernoulli NB e BNB semi-supervisionato (numpy) |
| `src/bayesian_network.py` | rete bayesiana TAN (numpy) |
| `src/prolog_kb.py` | esecuzione della KB (SWI-Prolog oppure motore Python equivalente) |
| `src/evaluation.py`, `src/visualization.py` | metriche, Friedman/Nemenyi, grafici |
| `tests/test_consistency.py` | coerenza OWL/Prolog/Python e correttezza dei modelli |

Poiché OWL, Prolog e feature derivano dalla stessa definizione, cambiare una soglia in
`src/knowledge.py` aggiorna tutto in modo coerente.

## 4. Conoscenza di background: ontologia OWL

### 4.1 Struttura

`wine.owl` contiene **52 classi** (37 definite tramite `equivalentClass`), **13 data property funzionali** e
**12 assiomi di disgiunzione**:

- `Vino` è la classe radice: un campione descritto dalle sue misure.
- 13 **classi-famiglia** raggruppano i concetti affini (`ProfiloAlcolico`, `ProfiloPH`, `ProfiloZuccherino`,
  `TipoVino`, `ProfiloEnologico`, ...) più `QualitaVino`.
- 34 **concetti definiti**: concetti di soglia, di tipo e composti.
- 3 **classi di qualità** (`QualitaBassa`, `QualitaMedia`, `QualitaAlta`) definite sul punteggio sensoriale.
- Data property: 11 misure (`xsd:decimal`), `tipoVino` (`xsd:string`), `punteggioQualita` (`xsd:integer`).
- I concetti mutuamente esclusivi di una famiglia sono dichiarati `AllDisjointClasses`.

### 4.2 Definizioni

Le classi sono **definite** (condizioni necessarie e sufficienti), quindi un reasoner può classificare un
individuo `Vino` nei concetti che soddisfa. In notazione di logica descrittiva:

```
AlcolMedio          ≡ Vino ⊓ ∃gradoAlcolico.[10.0 ≤ x < 12.0]
VinoRosso           ≡ Vino ⊓ ∃tipoVino.{"rosso"}
SO2TotaleEccessiva  ≡ (VinoRosso ⊓ SO2TotaleOltre150) ⊔ (VinoBianco ⊓ SO2TotaleOltre200)
RischioInstabilita  ≡ PHAlto ⊓ SO2LiberaBassa
VinoDifettoso       ≡ AciditaVolatileElevata ⊔ SO2TotaleEccessiva ⊔ RischioInstabilita
QualitaAlta         ≡ Vino ⊓ ∃punteggioQualita.[7, +∞)
```

### 4.3 Uso nella pipeline e coerenza

Nella pipeline l'appartenenza ai concetti è calcolata da `src/knowledge.evaluate_concepts` (veloce:
migliaia di vini per ciascuno dei run). È una funzione **deterministica delle misure, senza parametri stimati
dai dati**, quindi calcolarla su tutto il dataset non causa data leakage tra training e test.

Perché Python, OWL e Prolog dicano la stessa cosa, `tests/test_consistency.py` valuta i file generati con
interpreti minimali **indipendenti** (che leggono solo `wine.owl` e `wine_kb.pl`), anche su valori
*esattamente sulle soglie*. Il test usa un interprete minimale e non un reasoner completo: la verifica con
HermiT o Pellet si può fare aprendo `wine.owl` in Protégé.

### 4.4 Origine delle soglie

Le soglie sono di tre tipi, indicati nella colonna *Fondamento* della tabella:

- **normative o sensoriali**: classi di dolcezza UE per i vini fermi (`Secco`, `Abboccato`, `Amabile`),
  limiti UE sull'anidride solforosa totale per tipologia (versione semplificata, che ignora i limiti più alti
  per i vini con zuccheri ≥ 5 g/L), soglia sensoriale dell'acidità volatile (~0.6 g/L) e della SO2 libera (~50 mg/L);
- **convenzioni enologiche**: livelli di pH e di grado alcolico, SO2 libera insufficiente (< 15 mg/L);
- **empiriche**: acidità fissa, acido citrico, cloruri, densità, solfati.

Le soglie sono fissate **a priori** dalla conoscenza di dominio, senza ottimizzarle sulle etichette né sulle
partizioni di test. **Da fare prima della consegna:** citare le fonti puntuali (regolamenti UE, manuali di
enologia, OIV) per le soglie normative e convenzionali.

### 4.5 Catalogo dei concetti

<!-- CONCETTI:INIZIO -->
| Concetto | Definizione | Fondamento |
|---|---|---|
| `VinoRosso` | tipoVino = "rosso" | Tipologia indicata nel dataset. |
| `VinoBianco` | tipoVino = "bianco" | Tipologia indicata nel dataset. |
| `AciditaFissaBassa` | acidità fissa ∈ (-∞, 6.0) g/L | Soglia empirica: meno freschezza e minore stabilità. |
| `AciditaFissaAlta` | acidità fissa ∈ [8.5, +∞) g/L | Soglia empirica. |
| `AciditaVolatileBassa` | acidità volatile ∈ (-∞, 0.3) g/L | Acido acetico ben al di sotto della soglia di percezione. |
| `AciditaVolatileElevata` | acidità volatile ∈ [0.6, +∞) g/L | Oltre ~0.6 g/L l'acido acetico diventa percepibile (nota di aceto/solvente): soglia sensoriale di uso comune, inferiore al limite di legge. |
| `AcidoCitricoBasso` | acido citrico ∈ (-∞, 0.15) g/L | Soglia empirica: minore freschezza aromatica. |
| `AcidoCitricoElevato` | acido citrico ∈ [0.5, +∞) g/L | Soglia empirica. |
| `Secco` | zuccheri residui ∈ (-∞, 4.0] g/L | Indicazione di dolcezza UE: secco fino a 4 g/L (versione semplificata, senza la compensazione con l'acidità). |
| `Abboccato` | zuccheri residui ∈ (4.0, 12.0] g/L | Da 4 a 12 g/L (indicazione di dolcezza UE). |
| `Amabile` | zuccheri residui ∈ (12.0, +∞) g/L | Oltre 12 g/L: le classi UE 'amabile' (12-45) e 'dolce' (>45) sono unite perché i vini dolci sono poco rappresentati nel dataset. |
| `CloruriBassi` | cloruri ∈ (-∞, 0.05) g/L | Soglia empirica sul contenuto salino. |
| `CloruriElevati` | cloruri ∈ [0.1, +∞) g/L | Soglia empirica sul contenuto salino. |
| `SO2LiberaBassa` | SO2 libera ∈ (-∞, 15.0) mg/L | Protezione scarsa da ossidazione e microrganismi (soglia di uso comune). |
| `SO2LiberaAdeguata` | SO2 libera ∈ [15.0, 50.0] mg/L | Protezione adeguata senza note pungenti. |
| `SO2LiberaAlta` | SO2 libera ∈ (50.0, +∞) mg/L | Oltre ~50 mg/L la solforosa libera è percepibile all'olfatto (odore pungente). |
| `SO2TotaleOltre150` | SO2 totale ∈ (150.0, +∞) mg/L | Ausiliaria: limite UE per i vini rossi secchi. |
| `SO2TotaleOltre200` | SO2 totale ∈ (200.0, +∞) mg/L | Ausiliaria: limite UE per i vini bianchi secchi. |
| `SO2TotaleEccessiva` | (VinoRosso ∧ SO2TotaleOltre150) ∨ (VinoBianco ∧ SO2TotaleOltre200) | Limiti UE sull'anidride solforosa totale per vini secchi: 150 mg/L (rossi) e 200 mg/L (bianchi). Versione semplificata: ignora i limiti più alti previsti per i vini con zuccheri residui >= 5 g/L. |
| `DensitaBassa` | densità ∈ (-∞, 0.993) g/cm3 | L'alcol abbassa la densità. Soglia empirica. |
| `DensitaAlta` | densità ∈ [0.997, +∞) g/cm3 | Soglia empirica. |
| `PHBasso` | pH ∈ (-∞, 3.2) | Convenzione enologica. |
| `PHMedio` | pH ∈ [3.2, 3.5) | Convenzione enologica. |
| `PHAlto` | pH ∈ [3.5, +∞) | Con pH alto la solforosa è meno efficace. Convenzione enologica. |
| `SolfatiBassi` | solfati ∈ (-∞, 0.5) g/L | Soglia empirica (additivo con azione antimicrobica/antiossidante). |
| `SolfatiAlti` | solfati ∈ [0.75, +∞) g/L | Soglia empirica. |
| `AlcolBasso` | grado alcolico ∈ (-∞, 10.0) % vol | Convenzione: sotto 10 % vol. |
| `AlcolMedio` | grado alcolico ∈ [10.0, 12.0) % vol | Convenzione: tra 10 e 12 % vol. |
| `AlcolAlto` | grado alcolico ∈ [12.0, +∞) % vol | Convenzione: da 12 % vol, indice di uve molto mature. |
| `RischioInstabilita` | PHAlto ∧ SO2LiberaBassa | Un pH alto con SO2 libera bassa lascia il vino poco protetto (l'efficacia antimicrobica della solforosa cala al crescere del pH). |
| `VinoDifettoso` | AciditaVolatileElevata ∨ SO2TotaleEccessiva ∨ RischioInstabilita | Spunto acetico, eccesso di solforosa oppure rischio di instabilità. |
| `VinoFresco` | PHBasso ∧ (AlcolBasso ∨ AlcolMedio) ∧ AciditaVolatileBassa | Alta acidità (pH basso), alcol contenuto e assenza di spunto acetico: profilo tipico dei vini bianchi giovani e del vinho verde. |
| `VinoEquilibrato` | AlcolMedio ∧ AciditaVolatileBassa ∧ SO2LiberaAdeguata ∧ (PHBasso ∨ PHMedio) | Alcol medio, acidità volatile bassa, protezione solforosa adeguata e pH usuale. |
| `VinoMaturoEPulito` | AlcolAlto ∧ AciditaVolatileBassa | Grado alcolico alto (maturità delle uve) senza spunto acetico. |
<!-- CONCETTI:FINE -->

## 5. Knowledge Base Prolog

`kb/wine_kb.pl` è generata da `kb/build_kb.py` e contiene:

1. dichiarazioni dei **fatti** attesi, uno per vino: `wine(Id)`, `tipo(Id, rosso|bianco)` e un predicato
   per ogni misura (`grado_alcolico(Id, 12.5)`, `ph(Id, 3.2)`, ...);
2. **regole di derivazione dei concetti di soglia**, ad esempio
   `alcol_alto(V) :- grado_alcolico(V, X), X >= 12.0.`;
3. **regole dei concetti composti**, in forma di clausole di Horn, ad esempio
   `rischio_instabilita(V) :- ph_alto(V), so2_libera_bassa(V).` (le disgiunzioni diventano più clausole);
4. **regole di classificazione** e predicato di spiegazione.

### 5.1 Regole di classificazione

La classe è decisa dalla prima regola che scatta, in ordine di priorità: *bassa* → *alta* → *media* (default).

| Regola | Classe | Condizione | Motivazione |
|---|---|---|---|
| `difetto_enologico` | bassa | `VinoDifettoso` | un difetto analitico (spunto acetico, eccesso di SO2, rischio di instabilità) penalizza il vino |
| `poco_alcol_pochi_solfati` | bassa | `AlcolBasso` ∧ `SolfatiBassi` | poca struttura e poca protezione antiossidante |
| `maturo_e_pulito` | alta | `VinoMaturoEPulito` | uve mature (alcol alto) e fermentazione pulita |
| `equilibrato_con_solfati_alti` | alta | `VinoEquilibrato` ∧ `SolfatiAlti` | profilo equilibrato con buona protezione |
| *(default)* | media | — | nessuna regola applicabile |

Le regole sono **ipotesi a priori**: la KB non apprende e non vede mai le etichette. La copertura e la
precisione di ciascuna regola sono in `results/kb_rules.csv`.

### 5.2 Spiegabilità

`classifica(Vino, Classe, Motivo)` restituisce anche la regola che ha deciso. Esempio (fatti in fondo a `wine_kb.pl`):

```prolog
?- classifica(v1, Classe, Motivo).
Classe = alta, Motivo = maturo_e_pulito.
```

### 5.3 Esecuzione

Se `swipl` è nel PATH, `src/prolog_kb.py` scrive i fatti di tutti i vini in un file temporaneo insieme alla
KB e la interroga davvero in Prolog; in caso contrario usa il motore Python equivalente
(`src/knowledge.apply_rules`). Quando gira SWI-Prolog, l'accordo con il motore Python (atteso: 100 %) è un
controllo di coerenza riportato in `results/summary.md`.

## 6. Modelli

### 6.1 Classificatori supervisionati classici

Decision Tree (`max_depth=8`, `min_samples_leaf=5`), Random Forest (300 alberi, `min_samples_leaf=2`,
`max_features="sqrt"`) e Logistic Regression multinomiale (con standardizzazione, `C=1`), tutti con pesi di
classe bilanciati. Sono addestrati su due insiemi di feature: **Raw** (11 misure + tipo) e **Raw+OWL** (le
stesse più i 31 concetti binari). Con `--tune` gli iperparametri sono scelti con una CV a 3 fold interna al
solo training set.

### 6.2 Bernoulli Naive Bayes (BNB)

Per feature binarie `x` (i concetti OWL) e classe `c`:

```
P(c | x) ∝ P(c) · Π_j  θ_cj^(x_j) · (1 − θ_cj)^(1 − x_j)
θ_cj = (N_cj + α) / (N_c + 2α)            P(c) = (N_c + α) / (N + Kα)        (α = 1: Laplace)
```

Implementazione in numpy; il codice è verificato contro `sklearn.naive_bayes.BernoulliNB`.

### 6.3 BNB semi-supervisionato con EM (BNB-EM)

Si dispone di un insieme etichettato `L` (20 % del training, configurabile) e di uno non etichettato `U`
(il restante 80 %, di cui si ignora l'etichetta), come in Nigam et al. (2000):

```
inizializzazione:  M-step sui soli esempi di L
E-step:            γ_c(x) = P(c | x, θ)                    per x ∈ U
M-step:            N_c  = Σ_L 1[y=c]      + λ Σ_U γ_c(x)
                   N_cj = Σ_L 1[y=c] x_j  + λ Σ_U γ_c(x) x_j
convergenza:       variazione relativa della log-verosimiglianza < 1e-5
log L = Σ_L log P(x, y)  +  λ Σ_U log Σ_c P(x, c)
```

`λ ∈ [0, 1]` pesa i dati non etichettati. Il test automatico verifica che l'obiettivo (con il prior di
Laplace) non decresca mai e che l'EM migliori il BNB quando le ipotesi del modello valgono davvero.

### 6.4 Rete bayesiana TAN (Tree-Augmented Naive Bayes)

Come nel Naive Bayes la classe è genitore di ogni feature, ma le feature formano in più un **albero**: ogni
concetto ha al più un altro concetto come genitore. Questo permette di modellare le dipendenze che il Naive
Bayes ignora, ad esempio fra i concetti mutuamente esclusivi di una stessa famiglia (`AlcolBasso`,
`AlcolMedio`, `AlcolAlto`) o fra un concetto composto e i suoi componenti.

1. informazione mutua condizionata `I(X_i; X_j | C) = Σ_c P(c) Σ_{a,b} P(a,b|c) · log[ P(a,b|c) / (P(a|c) P(b|c)) ]`;
2. albero di copertura di peso massimo (algoritmo di Prim), orientato a partire da una radice;
3. tabelle di probabilità condizionata `P(X_j | C, X_parent(j))` con smoothing di Laplace;
4. inferenza esatta: `P(c | x) ∝ P(c) · Π_j P(x_j | c, x_parent(j))`.

La struttura appresa nel primo run è disegnata in `results/figures/tan_structure.png`.

### 6.5 Knowledge Base

Valutata sulle stesse righe di test degli altri modelli, senza alcun addestramento (sezione 5).

## 7. Protocollo sperimentale

- **Run e split**: 10 run con `StratifiedShuffleSplit` 70/30 (seed base 42, run *r* = 42 + *r*). Tutti i modelli
  vedono le stesse partizioni, condizione necessaria per il test di Friedman.
- **Semi-supervisionato**: nel training, il 20 % degli esempi (campionato in modo stratificato) mantiene
  l'etichetta; BNB usa solo quelli, BNB-EM usa anche gli altri senza etichetta. La **curva di apprendimento**
  ripete il confronto con l'1, 2, 5, 10, 20, 50 e 100 % di etichette e con due pesi `λ` dei non etichettati.
- **Metriche** (macro-medie): Precision, Recall, F1 e **G-Mean** = `(Π_k recall_k)^(1/K)`, la media geometrica
  delle recall per classe, che vale 0 se una classe non viene mai riconosciuta.
- **Confronto statistico** (Demšar, 2006): test di Friedman sulla F1 dei modelli nei run (blocchi = run) e,
  post-hoc di Nemenyi con differenza critica `CD = q_α · sqrt(k(k+1) / (6N))`, da interpretare solo se
  Friedman risulta significativo (`p < 0.05`).
- **Assenza di leakage**: concetti deterministici e senza parametri; standardizzazione dentro la pipeline di
  LR; tuning solo sul training; duplicati rimossi prima degli split.

## 8. Risultati

<!-- RISULTATI:INIZIO -->
_Questa sezione si popola con `python main.py --update-report` (tabelle in `results/summary.md`)._
<!-- RISULTATI:FINE -->

Grafici (cartella `results/figures/`):

| Figura | Contenuto |
|---|---|
| `class_distribution.png`, `correlation.png` | distribuzione delle classi e correlazioni tra misure, tipo e qualità |
| `owl_coverage.png` | % di vini di ciascuna classe che soddisfa ogni concetto |
| `metrics.png`, `f1_boxplot.png` | confronto tra modelli e variabilità sui run |
| `nemenyi.png` | p-value del post-hoc di Nemenyi |
| `confusion_*.png` | matrici di confusione (miglior modello e KB), somma sui run |
| `tan_structure.png` | struttura della rete bayesiana appresa |
| `semi_supervised_curve.png` | BNB vs BNB-EM al variare delle etichette |

## 9. Discussione

> **Da completare con i risultati reali.** Traccia di lettura per ciascuna domanda di ricerca.

- **RQ1 (concetti OWL come feature).** Confrontare DT/RF/LR con le versioni +OWL, per F1 e per rango medio.
  Se la differenza non è significativa al post-hoc di Nemenyi, i concetti non aggiungono informazione che i
  modelli non ricavino già dalle misure; per alberi e foreste, che apprendono da soli le soglie, è plausibile.
  Per la regressione logistica i concetti introducono non-linearità e potrebbero servire di più.
- **RQ2 (dati non etichettati).** Confrontare BNB e BNB-EM nella tabella e nella curva di apprendimento.
  L'EM massimizza la verosimiglianza del modello, non l'accuratezza: se le ipotesi del modello sono violate,
  i "cluster" trovati possono non coincidere con le classi (Cozman & Cohen, 2002). Qui l'indipendenza
  condizionale è violata di sicuro: i concetti di una famiglia sono mutuamente esclusivi e i composti sono
  funzioni di altri concetti. Ipotesi da verificare: la forte struttura rosso/bianco dei dati può dominare i
  cluster (controllare la composizione per tipo delle classi predette da BNB-EM) e un `λ` più basso dovrebbe
  attenuare il degrado.
- **RQ3 (TAN vs Naive Bayes).** Guardare `tan_structure.png`: archi fra concetti della stessa famiglia o
  fra un composto e i suoi componenti indicano che la rete sta correggendo l'indipendenza condizionale
  violata. Verificare se questo si traduce in un guadagno di F1 rispetto a BNB.
- **RQ4 (KB solo regole).** Leggere `kb_rules.csv` (quali regole sono precise, quali coprono pochi vini) e
  `confusion_KB.png` (quali classi la KB non riconosce). Il confronto con la G-Mean è istruttivo: una classe
  mai predetta azzera la G-Mean anche con una buona precisione.
- **F1 vs G-Mean.** Discutere dove le due metriche divergono e cosa dice sul bilanciamento tra classi.

## 10. Limiti e sviluppi futuri

- **Soglie**: alcune sono empiriche e non specifiche per tipo; si potrebbero distinguere soglie per rossi e
  bianchi e documentarle con fonti puntuali.
- **Tipo di vino noto**: il tipo è osservato per ogni vino, quindi nel Naive Bayes ed EM si potrebbe
  modellare `P(x | classe, tipo)` invece di `P(x | classe)`.
- **Qualità ordinale**: le classi sono ordinate; si potrebbero usare modelli e metriche ordinali.
- **Reasoner OWL**: integrare un reasoner (HermiT) con individui (ABox) per classificare i vini direttamente in OWL.
- **Generalizzazione**: un solo dataset, un solo prodotto (*vinho verde*); servirebbero altri vini.

## 11. Riproducibilità

```bash
pip install -r requirements.txt
python data/download_dataset.py
python main.py --update-report        # esperimenti, grafici e aggiornamento di questa relazione
python tests/test_consistency.py
```

Seed e iperparametri sono in `config.py`. Per fissare le versioni delle librerie usate: `pip freeze > requirements-lock.txt`.

## Riferimenti

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., Reis, J. (2009). *Modeling wine preferences by data mining
  from physicochemical properties.* Decision Support Systems 47(4), 547–553. UCI Machine Learning Repository,
  <https://doi.org/10.24432/C56S3T>.
- Nigam, K., McCallum, A., Thrun, S., Mitchell, T. (2000). *Text classification from labeled and unlabeled
  documents using EM.* Machine Learning 39(2–3), 103–134.
- Friedman, N., Geiger, D., Goldszmidt, M. (1997). *Bayesian network classifiers.* Machine Learning 29(2–3), 131–163.
- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple data sets.* Journal of Machine
  Learning Research 7, 1–30.
- Cozman, F., Cohen, I. (2002). *Unlabeled data can degrade classification performance of generative
  classifiers.* Proceedings of FLAIRS.
- Poole, D., Mackworth, A. (2023). *Artificial Intelligence: Foundations of Computational Agents* (3a ed.). Cambridge University Press.
- W3C (2012). *OWL 2 Web Ontology Language: Document Overview* (Second Edition).
- Lamy, J.-B. (2017). *Owlready: ontology-oriented programming in Python with automatic classification and
  high level constructs for biomedical ontologies.* Artificial Intelligence in Medicine 80, 11–28.
