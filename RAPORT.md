# Raport: heterogeniczność i kolejność nieliniowości w sieciach Forward-Forward

> Raport wypełniony według szablonu `RAPORT_SZABLON.md` (napisanego przed wynikami, PROTOKOL_v5 §9.4). Liczby przepisane bez zmian z `wyniki/WYNIKI_KONCOWE.json` (wyjście zamrożonego skryptu `ffexp/final.py`) i z plików predykcji `wyniki/test/` (hashe: `wyniki/HASHE_TEST.txt`). Zdania interpretacji w części 3 pochodzą wyłącznie z macierzy §7.4. Wszystko inne trafia do części 5 („Wyniki eksploracyjne”).
>
> Jednostki: dokładność w %, różnice Δ w punktach procentowych (p.p.). Δ = Acc(ramię testowane) − Acc(komparator), parowane po ziarnie.

**Status badania:** prerejestrowane z zewnętrznym znacznikiem czasu: https://github.com/slavek-is-hyperreal/ff-od-zera (tag `ff-od-zera-prereg`, push 2026-10-23 22:20 UTC; hashe checkpointów przed ewaluacją testową: commit `dc48e63`, push 2026-10-24 17:26 UTC); ukończone.
**Zakres wszystkich wniosków:** MLP, MNIST/Fashion-MNIST, schemat FF z §5 protokołu, na oficjalnym zbiorze testowym.

## Najważniejsze wyniki (pierwsza strona)

- **H1:** GORSZOŚĆ<S – LGT gorszy od co najmniej jednego bloku referencyjnego. Różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane. Komparatory istotnie lepsze od LGT: `K-GGT` (wyjście tanh; MNIST i F-MNIST), `K-R` (wyjście ReLU; tylko MNIST).
- **H2:** RÓWNOWAŻNOŚĆ+ – pozostałe kolejności równoważne LGT (±S). Wykryto różnicę istotną statystycznie, ale praktycznie pomijalną (< S); kierunek: MNIST −, F-MNIST − (co najmniej jedna inna kolejność nieznacznie lepsza od LGT).
- **H3:** GORSZOŚĆ<S – co najmniej jedna pojedyncza nieliniowość lepsza od banku (`P-R`, `P-G` na obu zbiorach). Na MNIST GORSZOŚĆ (różnica > S), na F-MNIST GORSZOŚĆ<S; łącznie kategoria słabsza. Różnica istotna statystycznie, mniejsza niż przyjęty próg S na F-MNIST; jej praktyczne znaczenie na obu zbiorach łącznie nie zostało wykazane.
- Wyniki nierozstrzygnięte lub niespójne: brak w analizie głównej. W analizie wrażliwości (a) (bootstrap 2D) H1 ma kategorię NIESPÓJNY-CZĘŚCIOWY (F-MNIST: NIEROZSTRZYGNIĘTY).
- Rozbieżność MNIST-10k vs QMNIST-50k (§7.8d): brak (H1 GORSZOŚĆ<S, H2 RÓWNOWAŻNOŚĆ+, H3 GORSZOŚĆ – jak na MNIST-10k).
- Odstępstwa od protokołu: 4 pozycje w `ODSTEPSTWA.md` (liczba procesów k; kolejność biegów w Etapach A/B/B2; podział Etapu C na 3 instancje; ewaluacja testowa na dwóch maszynach z kontrolą zgodności 0/960 000 różnic).

## 1. Wykonanie

| Element | Wartość |
|---|---|
| N (ziarna Etapu C) | 120 |
| S: H1/H2 – MNIST, F-MNIST | 0.30 p.p., 1.20 p.p. |
| S: H3 – MNIST, F-MNIST | 0.30 p.p., 1.35 p.p. |
| E_max (Etapy A, B, C) | 51, 51, 77 (reguła §5.2 uruchomiona po Etapie B+B2: 51 → 77) |
| Reguła brzegowa | uruchomiona: Etap B2, 456 biegów (szczegóły osi i punktów: `ETAP_B_WYBOR.json`, `plany/B2.json`) |
| GPU, obraz, wersje | NVIDIA RTX 3090 (Vast.ai, 3 instancje w Etapie C); `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`; Python 3.11.10, torch 2.5.1+cu124, numpy 2.1.2. Ewaluacja testowa: także CPU (torch 2.5.1+cpu, numpy 2.1.3) – zob. `ODSTEPSTWA.md` |
| Koszt łączny | ok. 10.2 USD z 38 USD (saldo 13.99 → 3.83 USD) |
| Biegi z NaN (per ramię) | 0 we wszystkich ramionach i zbiorach (tabela w części 2) |
| Odsetek biegów z aktywnym limitem epok | `P-T` MNIST: 21/120 (17.5%); `K-L` MNIST: 1/120; pozostałe ramiona: 0 (tabela w części 2) |

## 2. Dokładność testowa (średnia ± SD po N = 120 ziarnach)

| Ramię | MNIST | F-MNIST | NaN (M/F) | limit E_max (M/F) |
|---|---|---|---|---|
| `P-R` | 97.30 ± 0.18 | 86.53 ± 0.48 | 0/0 | 0/0 |
| `P-L` | 95.11 ± 0.45 | 83.60 ± 0.40 | 0/0 | 0/0 |
| `P-G` | 97.19 ± 0.18 | 86.10 ± 0.53 | 0/0 | 0/0 |
| `P-T` | 94.01 ± 1.02 | 83.85 ± 0.37 | 0/0 | 21/0 |
| `P-BANK` | 96.28 ± 0.28 | 85.26 ± 0.48 | 0/0 | 0/0 |
| `K-R` | 97.45 ± 0.22 | 87.38 ± 0.57 | 0/0 | 0/0 |
| `K-L` | 96.77 ± 0.31 | 86.08 ± 0.78 | 0/0 | 1/0 |
| `K-G` | 97.26 ± 0.25 | 87.35 ± 0.55 | 0/0 | 0/0 |
| `K-T` | 97.44 ± 0.28 | 87.00 ± 0.52 | 0/0 | 0/0 |
| `K-LLT` | 96.44 ± 0.37 | 85.90 ± 0.74 | 0/0 | 0/0 |
| `K-GGT` | 97.55 ± 0.22 | 87.56 ± 0.55 | 0/0 | 0/0 |
| `K-LGT` | 97.36 ± 0.29 | 87.31 ± 0.66 | 0/0 | 0/0 |
| `K-LTG` | 97.51 ± 0.21 | 86.49 ± 0.74 | 0/0 | 0/0 |
| `K-GLT` | 97.38 ± 0.24 | 87.33 ± 0.57 | 0/0 | 0/0 |
| `K-GTL` | 97.31 ± 0.29 | 87.03 ± 0.52 | 0/0 | 0/0 |
| `K-TLG` | 97.53 ± 0.26 | 87.38 ± 0.69 | 0/0 | 0/0 |
| `K-TGL` | 97.47 ± 0.21 | 87.57 ± 0.65 | 0/0 | 0/0 |
| `K-BANK` | 97.57 ± 0.24 | 87.59 ± 0.65 | 0/0 | 0/0 |

## 3. Hipotezy konfirmacyjne

Δ – średnia różnica parowana; PU – dwustronny przedział (1 − α′), α′ = 0.0167. Kategorie wg §7.3, łączenie wg §7.4.

### H1 (`K-LGT` vs `K-R`, `K-L`, `K-G`, `K-T`, `K-LLT`, `K-GGT`)

| Komparator | Δ MNIST [PU] | Δ F-MNIST [PU] |
|---|---|---|
| `K-R` | −0.090 [−0.172, −0.009] | −0.073 [−0.250, +0.105] |
| `K-L` | +0.591 [+0.497, +0.685] | +1.233 [+0.994, +1.471] |
| `K-G` | +0.095 [+0.012, +0.178] | −0.040 [−0.225, +0.145] |
| `K-T` | −0.086 [−0.175, +0.003] | +0.312 [+0.141, +0.483] |
| `K-LLT` | +0.919 [+0.810, +1.027] | +1.407 [+1.189, +1.624] |
| `K-GGT` | −0.196 [−0.274, −0.118] | −0.251 [−0.428, −0.074] |

Kategoria MNIST: GORSZOŚĆ<S; F-MNIST: GORSZOŚĆ<S; **łącznie: GORSZOŚĆ<S**. Równoważność: nie wykazana. Test Wilcoxona: GORSZOŚĆ<S na obu zbiorach.

### H2 (`K-LGT` vs `K-LTG`, `K-GLT`, `K-GTL`, `K-TLG`, `K-TGL`)

| Komparator | Δ MNIST [PU] | Δ F-MNIST [PU] |
|---|---|---|
| `K-LTG` | −0.150 [−0.233, −0.068] | +0.815 [+0.596, +1.034] |
| `K-GLT` | −0.022 [−0.108, +0.064] | −0.021 [−0.227, +0.185] |
| `K-GTL` | +0.053 [−0.039, +0.144] | +0.283 [+0.096, +0.469] |
| `K-TLG` | −0.169 [−0.251, −0.087] | −0.073 [−0.275, +0.129] |
| `K-TGL` | −0.116 [−0.191, −0.041] | −0.265 [−0.468, −0.062] |

Kategoria MNIST: RÓWNOWAŻNOŚĆ+; F-MNIST: RÓWNOWAŻNOŚĆ+; **łącznie: RÓWNOWAŻNOŚĆ+**. Równoważność (TOST, ±S): wykazana na obu zbiorach. Test Wilcoxona: RÓWNOWAŻNOŚĆ+ na obu zbiorach.

### H3 (`P-BANK` vs `P-R`, `P-L`, `P-G`, `P-T`)

| Komparator | Δ MNIST [PU] | Δ F-MNIST [PU] |
|---|---|---|
| `P-R` | −1.019 [−1.096, −0.942] | −1.268 [−1.411, −1.125] |
| `P-L` | +1.169 [+1.041, +1.297] | +1.664 [+1.523, +1.804] |
| `P-G` | −0.910 [−0.987, −0.834] | −0.838 [−0.997, −0.679] |
| `P-T` | +2.270 [+2.032, +2.507] | +1.414 [+1.278, +1.549] |

Kategoria MNIST: GORSZOŚĆ; F-MNIST: GORSZOŚĆ<S; **łącznie: GORSZOŚĆ<S**. Test Wilcoxona: GORSZOŚĆ (MNIST), GORSZOŚĆ<S (F-MNIST).

### Interpretacja (wyłącznie zdania z macierzy §7.4)

**H1 × H2 = G × R:** „LGT gorszy od co najmniej jednego bloku referencyjnego; pozostałe kolejności równoważne LGT (±S).”

**H3 = G:** „Co najmniej jedna pojedyncza nieliniowość lepsza od banku.”

**Obowiązkowe dopiski:**
- H1 (GORSZOŚĆ<S) i H3 (GORSZOŚĆ<S): „różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane”.
- H2 (RÓWNOWAŻNOŚĆ+): „wykryto różnicę istotną statystycznie, ale praktycznie pomijalną (< S)”, kierunek: MNIST −, F-MNIST −.
- Klasa G w H1 – komparatory, których PU (1 − α′) dla Δ leży całkowicie poniżej zera, wraz z nieliniowością wyjściową: `K-GGT` (tanh) – MNIST i F-MNIST; `K-R` (ReLU) – MNIST.

## 4. Analizy wrażliwości (§7.8)

| Hipoteza | Główny | (a) bootstrap 2D | (b) Wilcoxon | (c) biegi pierwotne | (d) QMNIST-50k (tylko MNIST) |
|---|---|---|---|---|---|
| H1 | GORSZOŚĆ<S | NIESPÓJNY-CZĘŚCIOWY (MNIST GORSZOŚĆ<S, F-MNIST NIEROZSTRZYGNIĘTY) | GORSZOŚĆ<S / GORSZOŚĆ<S | nie dotyczy (brak powtórek) | GORSZOŚĆ<S |
| H2 | RÓWNOWAŻNOŚĆ+ | RÓWNOWAŻNOŚĆ+ (MNIST RÓWNOWAŻNOŚĆ+, F-MNIST RÓWNOWAŻNOŚĆ) | RÓWNOWAŻNOŚĆ+ / RÓWNOWAŻNOŚĆ+ | nie dotyczy | RÓWNOWAŻNOŚĆ+ |
| H3 | GORSZOŚĆ<S | GORSZOŚĆ<S (MNIST GORSZOŚĆ, F-MNIST GORSZOŚĆ<S) | GORSZOŚĆ / GORSZOŚĆ<S | nie dotyczy | GORSZOŚĆ |

Twierdzenia o rozkładzie danych (a nie tylko o zbiorze testowym): dopuszczalne dla H2 i H3, gdzie (a) zgodne z głównym; niedopuszczalne dla H1.

---

## 5. Wyniki eksploracyjne – bez wniosków konfirmacyjnych

> W tej części wolno przypisywać mechanizmy, zawsze jako hipotezy do dalszego sprawdzenia. Brak korekty na wielokrotne porównania; przedziały to 95% CI.

- **E1 – głębokość (K vs P, ta sama nieliniowość; Δ = K − P, N = 120):**

  | Nieliniowość | MNIST | F-MNIST |
  |---|---|---|
  | R | +0.15 [+0.10, +0.20] | +0.85 [+0.73, +0.97] |
  | L | +1.66 [+1.56, +1.75] | +2.48 [+2.32, +2.64] |
  | G | +0.07 [+0.02, +0.13] | +1.25 [+1.10, +1.39] |
  | T | +3.44 [+3.25, +3.63] | +3.15 [+3.03, +3.27] |
  | BANK | +1.30 [+1.23, +1.36] | +2.32 [+2.19, +2.46] |

  Blok trzech podwarstw uczony lokalnie jest lepszy od warstwy płaskiej dla każdej nieliniowości; największy zysk dla tanh. Hipoteza do dalszego sprawdzenia: głębszy blok kompensuje problem ograniczonej dobroci tanh opisany dla płaskiego FF (Polar-FFA, `literatura/2408.09210_polar_ffa.md`).

- **E2 – luka do BP (Δ = BP − FF; N = 10; BP strojone w 2D, FF w 3D; §13.14):**

  | Para | MNIST | F-MNIST |
  |---|---|---|
  | BP-P-R − P-R | +0.62 [+0.52, +0.71] | +2.69 [+2.44, +2.95] |
  | BP-K-LGT − K-LGT | +0.28 [−0.01, +0.56] | +0.82 [−0.05, +1.69] |

  Dla bloku K-LGT luka FF–BP jest mniejsza niż dla sieci płaskiej i jej 95% CI obejmuje 0 na obu zbiorach.

- **E3 – kontrasty pojedyncze (wybrane; pełna lista w `WYNIKI_KONCOWE.json`):**
  - `K-BANK` vs bloki referencyjne: dodatni względem `K-LGT` (MNIST +0.22 [+0.15, +0.29], F-MNIST +0.28 [+0.10, +0.46]), `K-R`, `K-L`, `K-G`, `K-T`, `K-LLT`; względem `K-GGT` bez różnicy (MNIST +0.02 [−0.04, +0.08], F-MNIST +0.03 [−0.13, +0.19]). `K-BANK` ma najwyższą średnią dokładność ze wszystkich 18 ramion na obu zbiorach (część 2).
  - Inne kolejności vs `K-GGT`: żadna nie jest lepsza od `K-GGT` na F-MNIST; `K-LTG` na F-MNIST wyraźnie gorsze od bloków referencyjnych (vs `K-R`: −0.89 [−1.06, −0.71]).
  - Kontrast z bankiem w topologii płaskiej (`P-BANK` < `P-R`, `P-G`) i w topologii bloku (`K-BANK` ≥ wszystkie bloki) ma przeciwny znak. Hipoteza do sprawdzenia: bank szkodzi w pojedynczej warstwie FF (zgodnie z obserwacją Akkusa i in. o sumie kwadratów przy gęstych aktywacjach), ale nie w bloku uczonym lokalnie.
- **E4 – metryki drugorzędne:** nie wyliczone przez zamrożony skrypt `final.py`; do uzupełnienia jako osobna analiza eksploracyjna.
- **E5 – krajobraz strojenia:** nie wyliczony przez `final.py`; do uzupełnienia (dane: Etap A/B).
- **E6 – inferencja tylko z G₂ (dokładność %, N = 120):** najwyższe: `K-BANK` (MNIST 97.49, F-MNIST 87.44), `K-TLG` (97.48 / 87.21), `K-LTG` (MNIST 97.46), `K-GGT` (97.37 / 87.32); najniższe w blokach: `K-GLT` (94.88 / 83.65). Pełna tabela w `WYNIKI_KONCOWE.json`.
- **E7 – wyjście tanh (Δ = bloki z wyjściem tanh − pozostałe, N = 120):** MNIST −0.08 [−0.12, −0.05], F-MNIST +0.20 [+0.11, +0.29]. Kierunek przeciwny na dwóch zbiorach; obecność tanh na wyjściu bloku sama w sobie nie wyjaśnia różnic.
- **E8 – udział grup BANK w dobroci i gradiencie:** nie wyliczony przez `final.py`; do uzupełnienia.
- **E9 – przegląd szerokości (N = 10 na punkt; konfiguracje przeniesione z w = 250 bez ponownego strojenia, §13.12).** Przewidywanie autora zapisane przed badaniem: przewaga ramion heterogenicznych rośnie, gdy w maleje. Wynik (średnia dokładność testowa, %):

  | Ramię | MNIST w=32 | 64 | 128 | 250 | F-MNIST w=32 | 64 | 128 | 250 |
  |---|---|---|---|---|---|---|---|---|
  | `P-R` | 94.80 | 95.75 | 96.76 | 97.29 | 84.91 | 85.42 | 86.17 | 86.46 |
  | `P-BANK` | 92.32 | 94.05 | 95.50 | 96.11 | 76.95 | 82.49 | 84.74 | 85.48 |
  | `K-R` | 94.49 | 95.97 | 96.96 | 97.37 | 85.02 | 86.19 | 87.11 | 87.35 |
  | `K-LGT` | 94.14 | 95.67 | 96.76 | 97.33 | 84.91 | 85.98 | 86.78 | 87.24 |
  | `K-LLT` | 94.89 | 96.43 | 96.52 | 96.49 | 84.45 | 84.72 | 86.03 | 85.76 |
  | `K-GGT` | 94.82 | 96.57 | 97.18 | 97.56 | 85.25 | 86.46 | 87.65 | 87.54 |

  Różnica `K-LGT` minus najlepszy z bloków referencyjnych E9 przy w = 32 nie jest większa niż przy w = 250 (MNIST: −0.75 vs −0.23; F-MNIST: −0.34 vs −0.30); różnica `P-BANK` − `P-R` jest przy w = 32 bardziej ujemna niż przy w = 250 (MNIST −2.48 vs −1.18; F-MNIST −7.96 vs −0.98). **Uwaga:** „najlepszy blok referencyjny” wybrano tu wg średniej dokładności testowej; protokół (§7.7, E9) definiuje go wg średniej walidacyjnej z `zaslepione/` – opisowa różnica „w = 32 minus w = 250” z przedziałem bootstrap wg tej definicji nie jest wyliczana przez `final.py` i pozostaje do uzupełnienia. Kierunek wniosku nie zależy od wyboru komparatora: `K-LGT` jest przy w = 32 poniżej każdego z bloków referencyjnych E9 na MNIST, a na F-MNIST poniżej `K-R` i `K-GGT`. `K-GGT` ma najwyższą średnią przy w ≥ 64 na obu zbiorach. Pełne wartości z 95% CI w `WYNIKI_KONCOWE.json` (`E9_width_curves`). Rysunek 2: do wykonania.

- **E10 – spirala 2D (mediana dokładności testowej z 10 ziaren, w nawiasie liczba parametrów):**

  | Ramię | u=8 | u=16 | u=32 | u=64 | min. budżet z medianą ≥ 95% |
  |---|---|---|---|---|---|
  | `BP-P-R` | 97.5 (114) | 99.5 (354) | 99.5 (1218) | 99.5 (4482) | u=8 (114) |
  | `K-LGT` | 96.0 (120) | 97.5 (315) | 99.5 (1275) | 99.5 (4495) | u=8 (120) |
  | `K-GGT` | 94.7 (120) | 98.0 (315) | 99.5 (1275) | 99.5 (4495) | u=16 (315) |
  | `K-R` | 77.0 (120) | 89.2 (315) | 99.5 (1275) | 99.5 (4495) | u=32 (1275) |
  | `P-BANK` | 60.8 (112) | 82.3 (352) | 96.2 (1216) | 97.5 (4480) | u=32 (1216) |
  | `P-R` | 65.5 (112) | 96.8 (352) | 71.8 (1216) | 75.5 (4480) | u=16 (352) |
  | `P-T` | 61.0 (112) | 61.3 (352) | 61.5 (1216) | 71.0 (4480) | nie osiągnięto |

  Rysunek 1: `wyniki/E10/rysunek1_mapy_w16.png` (mapy decyzyjne). Na spirali bloki z gładkimi podwarstwami (`K-LGT`, `K-GGT`) osiągają ≥ 95% przy około 10 razy mniejszym budżecie niż `K-R`; na MNIST (E9) ten wzorzec dla `K-LGT` się nie powtórzył, dla `K-GGT` częściowo (najwyższa średnia przy w ≥ 64). Hipoteza do dalszego sprawdzenia: przewaga gładkich bloków przy małym budżecie zależy od zadania (gładka granica decyzyjna 2D vs obrazy).

- **Obserwacja nieplanowana (diagnostyka biegów):** w części biegów kryterium wczesnego zatrzymania warstwy 2 wybrało epokę 0 (np. `K-R` MNIST 36/120, `K-G` F-MNIST 45/120, `K-GLT` 24/19; zliczenia z `C/runs/*.json`, pole `best_epoch.layer2`). Do zbadania, czy warstwa 2 w tych ramionach wnosi coś do klasyfikacji (por. E6).

## 6. Ograniczenia

Pełna lista: PROTOKOL_v5 §13 (1–14) oraz odstępstwa z `ODSTEPSTWA.md` (4 pozycje). Szczególnie istotne dla interpretacji:
- §13.11: szeroki próg S na F-MNIST (1.20/1.35 p.p.) – równoważność w H2 na F-MNIST jest równoważnością w szerokim pasie.
- §13.12: w E9 konfiguracje przeniesione z w = 250 bez ponownego strojenia; mniejsze sieci mogły być dostrojone gorzej i nierówno między ramionami.
- E2 i E9 mają N = 10; E4, E5, E8 nie zostały wyliczone przez zamrożony skrypt analizy.
- Wnioski dotyczą jednej architektury (2 warstwy FF, w = 250), jednej funkcji dobroci (Σa²) i dwóch zbiorów danych.
