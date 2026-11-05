%% =====================================================================
%%  ENOKON - Knowledge Base Prolog sul dominio enologico
%%  FILE GENERATO da kb/build_kb.py a partire da src/knowledge.py:
%%  per modificarlo cambia src/knowledge.py e rigenera (python kb/build_kb.py).
%% =====================================================================

:- encoding(utf8).

%% --- 1. Fatti attesi (uno per vino) -----------------------------------
%%   wine(Id).                        identificativo del vino
%%   tipo(Id, rosso | bianco).
%%   acidita_fissa(Id, Valore).             % acidità fissa [g/L]
%%   acidita_volatile(Id, Valore).          % acidità volatile [g/L]
%%   acido_citrico(Id, Valore).             % acido citrico [g/L]
%%   zuccheri_residui(Id, Valore).          % zuccheri residui [g/L]
%%   cloruri(Id, Valore).                   % cloruri [g/L]
%%   so2_libera(Id, Valore).                % SO2 libera [mg/L]
%%   so2_totale(Id, Valore).                % SO2 totale [mg/L]
%%   densita(Id, Valore).                   % densità [g/cm3]
%%   ph(Id, Valore).                        % pH
%%   solfati(Id, Valore).                   % solfati [g/L]
%%   grado_alcolico(Id, Valore).            % grado alcolico [% vol]

:- dynamic wine/1, tipo/2, acidita_fissa/2, acidita_volatile/2, acido_citrico/2, zuccheri_residui/2, cloruri/2, so2_libera/2, so2_totale/2, densita/2, ph/2, solfati/2, grado_alcolico/2.

%% --- 2. Concetti di soglia e di tipo ----------------------------------
%% VinoRosso: tipoVino = "rosso"
vino_rosso(V) :- tipo(V, rosso).
%% VinoBianco: tipoVino = "bianco"
vino_bianco(V) :- tipo(V, bianco).
%% AciditaFissaBassa: acidità fissa ∈ (-∞, 6.0) g/L
acidita_fissa_bassa(V) :- acidita_fissa(V, X), X < 6.0.
%% AciditaFissaAlta: acidità fissa ∈ [8.5, +∞) g/L
acidita_fissa_alta(V) :- acidita_fissa(V, X), X >= 8.5.
%% AciditaVolatileBassa: acidità volatile ∈ (-∞, 0.3) g/L
acidita_volatile_bassa(V) :- acidita_volatile(V, X), X < 0.3.
%% AciditaVolatileElevata: acidità volatile ∈ [0.6, +∞) g/L
acidita_volatile_elevata(V) :- acidita_volatile(V, X), X >= 0.6.
%% AcidoCitricoBasso: acido citrico ∈ (-∞, 0.15) g/L
acido_citrico_basso(V) :- acido_citrico(V, X), X < 0.15.
%% AcidoCitricoElevato: acido citrico ∈ [0.5, +∞) g/L
acido_citrico_elevato(V) :- acido_citrico(V, X), X >= 0.5.
%% Secco: zuccheri residui ∈ (-∞, 4.0] g/L
secco(V) :- zuccheri_residui(V, X), X =< 4.0.
%% Abboccato: zuccheri residui ∈ (4.0, 12.0] g/L
abboccato(V) :- zuccheri_residui(V, X), X > 4.0, X =< 12.0.
%% Amabile: zuccheri residui ∈ (12.0, +∞) g/L
amabile(V) :- zuccheri_residui(V, X), X > 12.0.
%% CloruriBassi: cloruri ∈ (-∞, 0.05) g/L
cloruri_bassi(V) :- cloruri(V, X), X < 0.05.
%% CloruriElevati: cloruri ∈ [0.1, +∞) g/L
cloruri_elevati(V) :- cloruri(V, X), X >= 0.1.
%% SO2LiberaBassa: SO2 libera ∈ (-∞, 15.0) mg/L
so2_libera_bassa(V) :- so2_libera(V, X), X < 15.0.
%% SO2LiberaAdeguata: SO2 libera ∈ [15.0, 50.0] mg/L
so2_libera_adeguata(V) :- so2_libera(V, X), X >= 15.0, X =< 50.0.
%% SO2LiberaAlta: SO2 libera ∈ (50.0, +∞) mg/L
so2_libera_alta(V) :- so2_libera(V, X), X > 50.0.
%% SO2TotaleOltre150: SO2 totale ∈ (150.0, +∞) mg/L
so2_totale_oltre_150(V) :- so2_totale(V, X), X > 150.0.
%% SO2TotaleOltre200: SO2 totale ∈ (200.0, +∞) mg/L
so2_totale_oltre_200(V) :- so2_totale(V, X), X > 200.0.
%% DensitaBassa: densità ∈ (-∞, 0.993) g/cm3
densita_bassa(V) :- densita(V, X), X < 0.993.
%% DensitaAlta: densità ∈ [0.997, +∞) g/cm3
densita_alta(V) :- densita(V, X), X >= 0.997.
%% PHBasso: pH ∈ (-∞, 3.2)
ph_basso(V) :- ph(V, X), X < 3.2.
%% PHMedio: pH ∈ [3.2, 3.5)
ph_medio(V) :- ph(V, X), X >= 3.2, X < 3.5.
%% PHAlto: pH ∈ [3.5, +∞)
ph_alto(V) :- ph(V, X), X >= 3.5.
%% SolfatiBassi: solfati ∈ (-∞, 0.5) g/L
solfati_bassi(V) :- solfati(V, X), X < 0.5.
%% SolfatiAlti: solfati ∈ [0.75, +∞) g/L
solfati_alti(V) :- solfati(V, X), X >= 0.75.
%% AlcolBasso: grado alcolico ∈ (-∞, 10.0) % vol
alcol_basso(V) :- grado_alcolico(V, X), X < 10.0.
%% AlcolMedio: grado alcolico ∈ [10.0, 12.0) % vol
alcol_medio(V) :- grado_alcolico(V, X), X >= 10.0, X < 12.0.
%% AlcolAlto: grado alcolico ∈ [12.0, +∞) % vol
alcol_alto(V) :- grado_alcolico(V, X), X >= 12.0.

%% --- 3. Concetti composti (nozioni enologiche) --------------------------
%% SO2TotaleEccessiva := (VinoRosso ∧ SO2TotaleOltre150) ∨ (VinoBianco ∧ SO2TotaleOltre200)
so2_totale_eccessiva(V) :- vino_rosso(V), so2_totale_oltre_150(V).
so2_totale_eccessiva(V) :- vino_bianco(V), so2_totale_oltre_200(V).
%% RischioInstabilita := PHAlto ∧ SO2LiberaBassa
rischio_instabilita(V) :- ph_alto(V), so2_libera_bassa(V).
%% VinoDifettoso := AciditaVolatileElevata ∨ SO2TotaleEccessiva ∨ RischioInstabilita
vino_difettoso(V) :- acidita_volatile_elevata(V).
vino_difettoso(V) :- so2_totale_eccessiva(V).
vino_difettoso(V) :- rischio_instabilita(V).
%% VinoFresco := PHBasso ∧ (AlcolBasso ∨ AlcolMedio) ∧ AciditaVolatileBassa
vino_fresco(V) :- ph_basso(V), alcol_basso(V), acidita_volatile_bassa(V).
vino_fresco(V) :- ph_basso(V), alcol_medio(V), acidita_volatile_bassa(V).
%% VinoEquilibrato := AlcolMedio ∧ AciditaVolatileBassa ∧ SO2LiberaAdeguata ∧ (PHBasso ∨ PHMedio)
vino_equilibrato(V) :- alcol_medio(V), acidita_volatile_bassa(V), so2_libera_adeguata(V), ph_basso(V).
vino_equilibrato(V) :- alcol_medio(V), acidita_volatile_bassa(V), so2_libera_adeguata(V), ph_medio(V).
%% VinoMaturoEPulito := AlcolAlto ∧ AciditaVolatileBassa
vino_maturo_e_pulito(V) :- alcol_alto(V), acidita_volatile_bassa(V).

%% --- 4. Regole di classificazione della qualità ------------------------
%% regola_<classe>(Vino, Motivo): le regole sono provate in ordine di priorità
%% (bassa > alta > media: la prima che scatta decide).
%% Un vino con un difetto analitico (spunto acetico, eccesso di SO2, rischio di instabilità) è di qualità bassa.
regola_bassa(V, difetto_enologico) :- vino_difettoso(V).
%% Un vino povero di alcol e di solfati ha poca struttura e poca protezione: qualità bassa.
regola_bassa(V, poco_alcol_pochi_solfati) :- alcol_basso(V), solfati_bassi(V).
%% Un vino da uve mature e senza spunto acetico è di qualità alta.
regola_alta(V, maturo_e_pulito) :- vino_maturo_e_pulito(V).
%% Un vino equilibrato con solfati alti è di qualità alta.
regola_alta(V, equilibrato_con_solfati_alti) :- vino_equilibrato(V), solfati_alti(V).

%% classifica(+Vino, ?Classe, ?Motivo): classe di qualità e regola che l'ha decisa.
classifica(V, C, Motivo) :-
    (   regola_bassa(V, M) -> C = bassa, Motivo = M
    ;   regola_alta(V, M) -> C = alta, Motivo = M
    ;   C = media, Motivo = nessuna_regola
    ).

classifica(V, C) :- classifica(V, C, _).

%% --- Esempio d'uso (SWI-Prolog) -----------------------------------------
%%   ?- [wine_kb].
%%   ?- assertz(wine(v1)), assertz(tipo(v1, bianco)),
%%      assertz(grado_alcolico(v1, 12.5)), assertz(acidita_volatile(v1, 0.22)),
%%      assertz(acidita_fissa(v1, 6.8)), assertz(acido_citrico(v1, 0.3)),
%%      assertz(zuccheri_residui(v1, 2.0)), assertz(cloruri(v1, 0.04)),
%%      assertz(so2_libera(v1, 30.0)), assertz(so2_totale(v1, 110.0)),
%%      assertz(densita(v1, 0.991)), assertz(ph(v1, 3.2)), assertz(solfati(v1, 0.5)).
%%   ?- classifica(v1, Classe, Motivo).
%%   Classe = alta, Motivo = maturo_e_pulito.
