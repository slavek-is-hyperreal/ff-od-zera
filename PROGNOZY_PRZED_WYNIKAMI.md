# Prognozy Claude'a przed odślepieniem (zabawa, nie część protokołu)

Zapisane 2026-10-24 ok. 20:20 CEST, w trakcie ewaluacji testowej, przed uruchomieniem `ffexp.final`. Podstawa: walidacja Etapu B, piloty, literatura (Polar-FFA, Akkus i in.). Wyniki Etapu C nieznane (zaślepione).

| Hipoteza | Prognoza (MNIST / F-MNIST → łącznie) | Pewność |
|---|---|---|
| H1 (K-LGT vs 6 bloków ref.) | GORSZOŚĆ<S / RÓWNOWAŻNOŚĆ lub NIEROZSTRZYGNIĘTY → NIESPÓJNY-CZĘŚCIOWY | brak PRZEWAGI: 90%; dokładna kategoria łączna: 30% |
| H2 (LGT vs 5 permutacji) | NIEROZSTRZYGNIĘTY (wygrana z GTL/TGL, remis z GLT) | 55% |
| H3 (P-BANK vs 4 płaskie) | GORSZOŚĆ / GORSZOŚĆ<S → GORSZOŚĆ<S (klasa G) | klasa G: 60% |
| E9 | przewaga LGT nad najlepszym ref. rośnie przy mniejszym w | 55% |
| Zgodność CPU/GPU | > 99.9% przykładów | 85% |

## Prognoza autora (Sławek), ok. 20:22 CEST

„Stawiam, że wyjdzie nam E9 jedynie”: żadna z hipotez konfirmacyjnych H1–H3 nie da pozytywnego wyniku (przewagi), a jedynym „wynikiem” będzie efekt w eksploracyjnym E9 (przewaga ramion heterogenicznych rośnie przy mniejszej szerokości).

## Rozliczenie (po `ffexp.final`, 2026-10-24 ok. 21:10)

Wyniki konfirmacyjne (łącznie; MNIST / F-MNIST): H1 GORSZOŚĆ<S (GORSZOŚĆ<S / GORSZOŚĆ<S); H2 RÓWNOWAŻNOŚĆ+ (RÓWNOWAŻNOŚĆ+ / RÓWNOWAŻNOŚĆ+); H3 GORSZOŚĆ<S (GORSZOŚĆ / GORSZOŚĆ<S). E9: przewidywanie autora protokołu (przewaga ramion heterogenicznych rośnie przy mniejszym w) nie znalazło wsparcia – strata K-LGT i P-BANK względem najlepszego ramienia referencyjnego przy w = 32 jest taka sama albo większa niż przy w = 250.

| Prognoza | Trafiona? |
|---|---|
| Claude H1: brak PRZEWAGI | tak |
| Claude H1: MNIST GORSZOŚĆ<S | tak |
| Claude H1: łącznie NIESPÓJNY-CZĘŚCIOWY | nie (GORSZOŚĆ<S; NIESPÓJNY-CZĘŚCIOWY wyszedł tylko w analizie wrażliwości z bootstrapem) |
| Claude H2: NIEROZSTRZYGNIĘTY, GTL/TGL słabe | nie (RÓWNOWAŻNOŚĆ+; TGL nawet nieco lepsze od LGT) |
| Claude H3: GORSZOŚĆ / GORSZOŚĆ<S → GORSZOŚĆ<S | tak, co do kategorii na obu zbiorach i łącznie |
| Claude E9: przewaga LGT rośnie przy małym w | nie |
| Claude: zgodność CPU/GPU > 99.9% | tak (100%, 0/960 000) |
| Sławek: H1–H3 bez przewagi | tak |
| Sławek: „wyjdzie tylko E9” | nie (E9 nie wsparło przewidywania) |
