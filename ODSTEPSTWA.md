# Odstępstwa od protokołu po zamrożeniu (tag `ff-od-zera-prereg`)

Każdy wpis: data, treść, powód, czy przed czy po ewaluacji testowej (§10.6).

| Data | Zmiana | Powód | Przed/po ewaluacji testowej |
|---|---|---|---|
| 2026-10-24 | Etap A: liczba procesów równoległych k = 16 → 12 (D6 / §16 podawały k = 16); biegi przerwane błędem braku pamięci GPU (OutOfMemoryError przy alokacji bufora warstwy 1) uruchomione ponownie od zera z tym samym ziarnem (§8) | test przepustowości (D6) nie odtworzył szczytowego zużycia pamięci prawdziwego runnera (oba zbiory danych na GPU + bufor); przy k = 16 ~9% biegów kończyło się błędem OOM. k nie wpływa na wynik pojedynczego biegu (każdy bieg w osobnym, deterministycznym procesie) | przed (trening Etapu A) |
| 2026-10-24 | §8: w Etapach A, B i B2 biegi wykonywano w kolejności planu (zbiór → punkt/ziarno → ramię), a nie w losowej kolejności. Od Etapu C plan jest permutowany ziarnem `seed_from("C-order")` | przeoczenie implementacyjne. Każdy bieg jest deterministyczny i wykonywany w osobnym procesie, więc kolejność nie wpływa na jego wynik; dotyczy wyłącznie ochrony przed zmianami warunków maszyny w czasie | przed (trening) |
| 2026-10-24 | Etap C podzielony na dwie połowy planu (`C1.json`, `C2.json`, losowa permutacja, ten sam typ GPU) dla ewentualnego równoległego wykonania na dwóch instancjach RTX 3090 | prognoza czasu Etapu C wzrosła po podniesieniu E_max do 77 (reguła §5.2). §8 zabrania zmiany typu GPU w trakcie etapu, nie liczby instancji tego samego typu | przed (trening) |
