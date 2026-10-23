# Odstępstwa od protokołu po zamrożeniu (tag `ff-od-zera-prereg`)

Każdy wpis: data, treść, powód, czy przed czy po ewaluacji testowej (§10.6).

| Data | Zmiana | Powód | Przed/po ewaluacji testowej |
|---|---|---|---|
| 2026-10-24 | Etap A: liczba procesów równoległych k = 16 → 12 (D6 / §16 podawały k = 16); biegi przerwane błędem braku pamięci GPU (OutOfMemoryError przy alokacji bufora warstwy 1) uruchomione ponownie od zera z tym samym ziarnem (§8) | test przepustowości (D6) nie odtworzył szczytowego zużycia pamięci prawdziwego runnera (oba zbiory danych na GPU + bufor); przy k = 16 ~9% biegów kończyło się błędem OOM. k nie wpływa na wynik pojedynczego biegu (każdy bieg w osobnym, deterministycznym procesie) | przed (trening Etapu A) |
