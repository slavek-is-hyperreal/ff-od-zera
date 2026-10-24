# Raport: heterogeniczność i kolejność nieliniowości w sieciach Forward-Forward

> **Wersja 2** (po recenzji `recenzje/R6_raport.md`). **Status: wersja robocza z jawną listą braków** – nie wykonano E4, E5 (obowiązkowe wg §7.7), E8 ani Rysunku 2; szczegóły w części 6.
>
> Raport wypełniony według szablonu `RAPORT_SZABLON.md` (napisanego przed wynikami, PROTOKOL_v5 §9.4). Źródła liczb: `wyniki/WYNIKI_KONCOWE.json` (wyjście zamrożonego skryptu `ffexp/final.py`, przepisane bez zmian) oraz – tam, gdzie zaznaczono „[obliczone poza `final.py`]” – obliczenia z plików predykcji `wyniki/test/` (hashe: `wyniki/HASHE_TEST.txt`), rekordów biegów i plików `zaslepione/` (dostępnych po kroku 2 §5.6). Zdania interpretacji w części 3 pochodzą wyłącznie z macierzy §7.4. Wszystko inne trafia do części 5 („Wyniki eksploracyjne”).
>
> Jednostki: dokładność w %, różnice Δ w punktach procentowych (p.p.). Δ = Acc(ramię testowane) − Acc(komparator), parowane po ziarnie.

**Status badania:** prerejestrowane z zewnętrznym znacznikiem czasu: https://github.com/slavek-is-hyperreal/ff-od-zera (tag `ff-od-zera-prereg`, push 2026-10-23 22:20 UTC; hashe checkpointów przed ewaluacją testową: commit `4b26d2c` w repozytorium roboczym = `dc48e63` w publicznej gałęzi, push 2026-10-24 17:26 UTC); ukończone.
**Zakres wszystkich wniosków:** MLP, MNIST/Fashion-MNIST, schemat FF z §5 protokołu, na oficjalnym zbiorze testowym.

## Najważniejsze wyniki (pierwsza strona)

**Przewidywania autora z §2 (przewaga ramion heterogenicznych / kolejności LGT) nie potwierdziły się w żadnej z trzech hipotez. W H1 i H3 wynik ma kierunek przeciwny do przewidywanego (klasa G).**


- **H1:** GORSZOŚĆ<S – LGT gorszy od co najmniej jednego bloku referencyjnego. Różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane. Jedyny komparator istotnie lepszy od LGT (procedura Holma, §7.3): `K-GGT` (wyjście tanh), na MNIST i F-MNIST.
- **H2:** RÓWNOWAŻNOŚĆ+ – pozostałe kolejności równoważne LGT (±S). Wykryto różnicę istotną statystycznie, ale praktycznie pomijalną (< S); kierunek: MNIST −, F-MNIST −. Kierunek wyznaczają kolejności istotnie lepsze od LGT (Holm): MNIST – `K-TLG`, `K-LTG`, `K-TGL`; F-MNIST – `K-TGL`. Jednocześnie na F-MNIST LGT jest istotnie lepszy od `K-LTG` i `K-GTL` (wszystko w granicach ±S).
- **H3:** GORSZOŚĆ<S – co najmniej jedna pojedyncza nieliniowość lepsza od banku (Holm: `P-R`, `P-G` na obu zbiorach). Na MNIST GORSZOŚĆ, na F-MNIST GORSZOŚĆ<S; łączna kategoria jest słabszą z dwóch (§7.4). „Różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane.” (Na F-MNIST oznacza to: nie wykazano różnicy większej niż S = 1.35 p.p.; średnia Δ względem `P-R` wynosi −1.27, a jej PU obejmuje −S.)
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
| Reguła brzegowa | uruchomiona: Etap B2, 456 biegów (osie i punkty: `ETAP_B_WYBOR.json`, `plany/B2.json`; wg recenzji R6: osie wd i c) |
| GPU, obraz, wersje | NVIDIA RTX 3090 (Vast.ai, 3 instancje w Etapie C, różne hosty; wersje sterowników per bieg w polu `versions` rekordów `C/runs/*.json`); `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`; Python 3.11.10, torch 2.5.1+cu124 (numpy 2.1.2 wg sprawdzenia środowiska instancji przy wysyłce kodu, nie z rekordów biegów); kod Etapu C: commit `8df4f0f` (pole `versions.code_commit` w rekordach biegów). Ewaluacja testowa: także CPU (torch 2.5.1+cpu, numpy 2.1.3) – zob. `ODSTEPSTWA.md` |
| Koszt łączny | ok. 10.2 USD z 38 USD (różnica salda konta Vast.ai przed Etapem A i po zniszczeniu ostatniej instancji: 13.99 → 3.83 USD; bez faktur) |
| Biegi z NaN (per ramię) | 0 we wszystkich ramionach i zbiorach (tabela w części 2) |
| Odsetek biegów z aktywnym limitem epok | wg definicji §5.2 (najlepsza epoka ≥ E_max − 2 w którejkolwiek warstwie, epoki indeksowane od 0): `P-T` MNIST 11/120 (9.2%), pozostałe ramiona 0. Dodatkowo: trening do E_max (wyczerpanie limitu) – `P-T` MNIST 21/120, `K-L` MNIST 1/120 [obliczone poza `final.py`] |

## 2. Dokładność testowa (średnia ± SD po N = 120 ziarnach)

| Ramię | MNIST | F-MNIST | NaN (M/F) | trening do E_max (M/F) |
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
- Klasa G w H1 – komparatory istotnie lepsze od LGT (jednostronne p dla H₀: Δ_K ≥ 0, procedura Holma na poziomie α′/2 = 0.0083, jak w warunku inf(0) §7.3) [obliczone poza `final.py`, który zwraca tylko kategorię]: `K-GGT` (wyjście tanh) – MNIST (p = 6.1·10⁻⁹) i F-MNIST (p = 4.0·10⁻⁴). `K-R` na MNIST nie przechodzi procedury Holma (p = 3.9·10⁻³ > 0.0083/5), choć jego PU (1 − α′) leży poniżej zera.
- Testy Wilcoxona raportowane są wyłącznie jako kategorie (tak liczy `final.py`); przedziały Hodgesa–Lehmanna nie były wyliczane.

## 4. Analizy wrażliwości (§7.8)

| Hipoteza | Główny | (a) bootstrap 2D | (b) Wilcoxon | (c) biegi pierwotne | (d) QMNIST-50k (tylko MNIST) |
|---|---|---|---|---|---|
| H1 | GORSZOŚĆ<S | NIESPÓJNY-CZĘŚCIOWY (MNIST GORSZOŚĆ<S, F-MNIST NIEROZSTRZYGNIĘTY) | GORSZOŚĆ<S / GORSZOŚĆ<S | nie dotyczy (brak powtórek) | GORSZOŚĆ<S |
| H2 | RÓWNOWAŻNOŚĆ+ | RÓWNOWAŻNOŚĆ+ (MNIST RÓWNOWAŻNOŚĆ+, F-MNIST RÓWNOWAŻNOŚĆ) | RÓWNOWAŻNOŚĆ+ / RÓWNOWAŻNOŚĆ+ | nie dotyczy | RÓWNOWAŻNOŚĆ+ |
| H3 | GORSZOŚĆ<S | GORSZOŚĆ<S (MNIST GORSZOŚĆ, F-MNIST GORSZOŚĆ<S) | GORSZOŚĆ / GORSZOŚĆ<S | nie dotyczy | GORSZOŚĆ |

Twierdzenia o rozkładzie danych (a nie tylko o zbiorze testowym): dopuszczalne dla H3 (GORSZOŚĆ<S) oraz dla H2 tylko w zakresie równoważności (±S) – w (a) na F-MNIST kategoria to RÓWNOWAŻNOŚĆ, bez istotnej różnicy; niedopuszczalne dla H1. (c) nie dotyczy: w Etapie C nie było powtórek po błędzie kodu (§5.6 pkt 4), więc brak biegów pierwotnych.

---

## 5. Wyniki eksploracyjne – bez wniosków konfirmacyjnych

> W tej części wolno przypisywać mechanizmy, zawsze jako hipotezy do dalszego sprawdzenia. Brak korekty na wielokrotne porównania; przedziały to 95% CI.

- **E1 – głębokość (K vs P, ta sama nieliniowość; Δ = K − P, N = 120):**

  | Nieliniowość | MNIST | F-MNIST |
  |---|---|---|
  | R | +0.15 [+0.10, +0.20] | +0.85 [+0.73, +0.97] |
  | L | +1.66 [+1.56, +1.75] | +2.48 [+2.32, +2.64] |
  | G | +0.07 [+0.02, +0.13] | +1.25 [+1.10, +1.39] |
  | T | +3.44 [+3.25, +3.63] | +3.15 [+3.03, +3.26] |
  | BANK | +1.30 [+1.23, +1.36] | +2.32 [+2.19, +2.46] |

  Blok trzech podwarstw uczony lokalnie ma wyższą dokładność niż warstwa płaska dla każdej nieliniowości; największa różnica dla tanh. Hipoteza do dalszego sprawdzenia: dodatkowe podwarstwy przed wyjściem tanh łagodzą problem niskiej wariancji normy przy ograniczonym wyjściu opisany dla płaskiego FF (Polar-FFA, `literatura/2408.09210_polar_ffa.md`) – dobroć `K-T` pozostaje ograniczona (G ≤ w), więc nie chodzi o samo zniesienie ograniczenia.

- **E2 – luka do BP (Δ = BP − FF; N = 10; BP strojone w 2D, FF w 3D; §13.14):**

  | Para | MNIST | F-MNIST |
  |---|---|---|
  | BP-P-R − P-R | +0.62 [+0.52, +0.71] | +2.69 [+2.44, +2.95] |
  | BP-K-LGT − K-LGT | +0.28 [−0.01, +0.56] | +0.82 [−0.05, +1.69] |

  Dla bloku K-LGT luka FF–BP jest mniejsza niż dla sieci płaskiej; jej 95% CI obejmuje 0 na obu zbiorach, co przy N = 10 nie oznacza braku luki. Część różnicy wynika z punktu odniesienia: `BP-K-LGT` jest słabszy od `BP-P-R` (MNIST 97.60 vs 97.90, F-MNIST 88.07 vs 89.15; średnie z 10 ziaren) [obliczone poza `final.py`].

- **E3 – kontrasty pojedyncze (wybrane; pełna lista w `WYNIKI_KONCOWE.json`):**
  - `K-BANK` vs bloki referencyjne: dodatni względem `K-LGT` (MNIST +0.22 [+0.15, +0.29], F-MNIST +0.28 [+0.10, +0.46]), `K-R`, `K-L`, `K-G`, `K-T`, `K-LLT`; względem `K-GGT` bez różnicy (MNIST +0.02 [−0.04, +0.08], F-MNIST +0.03 [−0.13, +0.19]). `K-BANK` ma najwyższą średnią na obu zbiorach, ale przewaga nad `K-GGT` (0.02–0.03 p.p.) mieści się w szumie.
  - Inne kolejności vs `K-GGT`: na F-MNIST żadna nie jest istotnie lepsza od `K-GGT` (`K-TGL` +0.01 [−0.13, +0.15]). `K-LTG` na F-MNIST jest gorsze od `K-R`, `K-G`, `K-T`, `K-GGT` (np. vs `K-R`: −0.89 [−1.06, −0.71]), ale lepsze od `K-L` i `K-LLT`.
  - Kontrast banku z jednorodnymi nieliniowościami ma przeciwny znak w topologii płaskiej (`P-BANK` < `P-R`, `P-G`) i w topologii bloku (`K-BANK` ≥ każdy blok jednorodny). Hipoteza do sprawdzenia: w pojedynczej warstwie FF bank z jednostkami o różnych skalach aktywacji zaburza sumę kwadratów jako dobroć, a blok uczony lokalnie to kompensuje. (Wyjaśnienie „gęste aktywacje szkodzą Σa²” z pracy Akkusa i in. nie pasuje do dobrego wyniku `P-G`, więc nie jest tu przyjmowane.)
- **E4 – metryki drugorzędne:** nie wyliczone przez zamrożony skrypt `final.py`; do uzupełnienia jako osobna analiza eksploracyjna.
- **E5 – krajobraz strojenia:** nie wyliczony przez `final.py`; do uzupełnienia (dane: Etap A/B).
- **E6 – inferencja tylko z G₂ (dokładność %, N = 120; MNIST / F-MNIST):** `K-BANK` 97.49 / 87.44, `K-TLG` 97.48 / 87.21, `K-LTG` 97.46 / 86.05, `K-GGT` 97.37 / 87.32, `K-LGT` 97.25 / 87.06; najniższe w blokach: `K-GLT` 94.88 / 83.65. Pełna tabela w `WYNIKI_KONCOWE.json`.
- **E7 – wyjście tanh (N = 120):** średnia różnica bloki z wyjściem tanh − pozostałe: MNIST −0.08 [−0.12, −0.05], F-MNIST +0.20 [+0.11, +0.29]; LGT − GLT: MNIST −0.02 [−0.09, +0.05], F-MNIST −0.02 [−0.19, +0.15]; LGT − `K-T`: −0.09 [−0.16, −0.01] / +0.31 [+0.17, +0.45]; LGT − `K-LLT`: +0.92 [+0.83, +1.01] / +1.41 [+1.23, +1.58]; LGT − `K-GGT`: −0.20 [−0.26, −0.13] / −0.25 [−0.40, −0.11]. Hipoteza: kierunek efektu wyjścia tanh zależy od zbioru, więc sama obecność tanh na wyjściu bloku nie wyjaśnia różnic między blokami.
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

  Kolumna w = 250 to podzbiór 10 ziaren (10001–10010) z Etapu C, stąd różnice względem części 2 (np. `K-LGT` MNIST 97.33 vs 97.36).

  **Kontrast z protokołu (§7.7)** [obliczone poza `final.py`]: najlepsze ramię referencyjne wybrane wg średniej dokładności walidacyjnej (`zaslepione/`) – `K-GGT` przy każdym w na obu zbiorach. Różnica (w = 32) − (w = 250) z 95% przedziałem bootstrap (20 000 losowań ziaren, różnice sparowane po ziarnie):

  | Kontrast | MNIST | F-MNIST |
  |---|---|---|
  | `K-LGT` − `K-GGT`: w = 32 / w = 250 | −0.68 / −0.23 | −0.34 / −0.30 |
  | (w = 32) − (w = 250) | −0.45 [−1.15, +0.35] | −0.04 [−1.05, +0.90] |
  | `P-BANK` − `P-R`: w = 32 / w = 250 | −2.48 / −1.17 | −7.96 / −0.98 |
  | (w = 32) − (w = 250) | −1.31 [−1.70, −0.88] | −6.98 [−9.41, −4.60] |

  Wynik: dla `K-LGT` dane przy N = 10 nie rozstrzygają przewidywania (przedział obejmuje 0); dla `P-BANK` różnica przy w = 32 jest bardziej ujemna niż przy w = 250, przeciwnie do przewidywania.

  Różnice względem każdego ramienia E9 (średnia [95% CI t], N = 10):

  | Kontrast | MNIST w=32 | 64 | 128 | 250 |
  |---|---|---|---|---|
  | `K-LGT` − `K-R` | −0.35 [−1.00, +0.31] | −0.30 [−0.58, −0.01] | −0.20 [−0.45, +0.05] | −0.05 [−0.29, +0.19] |
  | `K-LGT` − `K-LLT` | −0.74 [−1.43, −0.06] | −0.76 [−1.26, −0.26] | +0.24 [−0.04, +0.53] | +0.84 [+0.54, +1.14] |
  | `K-LGT` − `K-GGT` | −0.68 [−1.54, +0.18] | −0.90 [−1.24, −0.55] | −0.43 [−0.76, −0.09] | −0.23 [−0.47, −0.00] |
  | `P-BANK` − `P-R` | −2.48 [−2.97, −1.99] | −1.70 [−2.23, −1.17] | −1.26 [−1.56, −0.96] | −1.17 [−1.33, −1.01] |

  | Kontrast | F-MNIST w=32 | 64 | 128 | 250 |
  |---|---|---|---|---|
  | `K-LGT` − `K-R` | −0.11 [−0.73, +0.51] | −0.21 [−0.82, +0.41] | −0.33 [−0.96, +0.31] | −0.10 [−0.87, +0.67] |
  | `K-LGT` − `K-LLT` | +0.46 [−0.26, +1.19] | +1.25 [+0.32, +2.19] | +0.75 [+0.03, +1.47] | +1.48 [+0.60, +2.36] |
  | `K-LGT` − `K-GGT` | −0.34 [−1.27, +0.59] | −0.48 [−0.91, −0.05] | −0.87 [−1.48, −0.27] | −0.30 [−1.01, +0.41] |
  | `P-BANK` − `P-R` | −7.96 [−10.96, −4.97] | −2.93 [−4.62, −1.24] | −1.43 [−1.89, −0.96] | −0.98 [−1.27, −0.69] | `K-GGT` ma najwyższą średnią przy w ≥ 64 na obu zbiorach. Pełne wartości z 95% CI w `WYNIKI_KONCOWE.json` (`E9_width_curves`). Rysunek 2: do wykonania.

- **E10 – spirala 2D (mediana dokładności testowej z 10 ziaren, w nawiasie liczba parametrów; w – szerokość/budżet z §3.4):**

  | Ramię | w=8 | w=16 | w=32 | w=64 | min. budżet z medianą ≥ 95% |
  |---|---|---|---|---|---|
  | `BP-P-R` | 97.5 (114) | 99.5 (354) | 99.5 (1218) | 99.5 (4482) | w=8 (114) |
  | `K-LGT` | 96.0 (120) | 97.5 (315) | 99.5 (1275) | 99.5 (4495) | w=8 (120) |
  | `K-GGT` | 94.7 (120) | 98.0 (315) | 99.5 (1275) | 99.5 (4495) | w=16 (315) |
  | `K-R` | 77.0 (120) | 89.2 (315) | 99.5 (1275) | 99.5 (4495) | w=32 (1275) |
  | `P-BANK` | 60.8 (112) | 82.3 (352) | 96.2 (1216) | 97.5 (4480) | w=32 (1216) |
  | `P-R` | 65.5 (112) | 96.8 (352) | 71.8 (1216) | 75.5 (4480) | w=16 (352) |
  | `P-T` | 61.0 (112) | 61.3 (352) | 61.5 (1216) | 71.0 (4480) | nie osiągnięto |

  Rysunek 1: `wyniki/E10/rysunek1_mapy_w16.png` – mapy decyzyjne przy w = 16, dla każdego ramienia ziarno o dolnej medianie dokładności walidacyjnej: `P-R` ziarno 50004 (352 param., test 97.5%), `P-T` 50009 (352, 61.5%), `P-BANK` 50003 (352, 80.0%), `K-R` 50010 (315, 89.0%), `K-LGT` 50010 (315, 99.0%), `K-GGT` 50001 (315, 100.0%), `BP-P-R` 50005 (354, 100.0%).

  Mediana ≥ 95% osiągnięta przez `K-LGT` przy ok. 10 razy mniejszym budżecie niż `K-R` (120 vs 1275 param.), przez `K-GGT` przy ok. 4 razy mniejszym (315 vs 1275). Wg średniej (nie mediany) `K-LGT` nie osiąga 95% przy w = 8 (91.4) i w = 16 (93.9). `P-R` jest niemonotoniczny względem budżetu (mediana 96.8 przy w = 16, 71.8 przy w = 32), co wskazuje na niestabilność optymalizacji płaskiego FF na spirali. Na MNIST (E9) wzorzec dla `K-LGT` się nie powtórzył; `K-GGT` ma tam najwyższą średnią przy w ≥ 64. Hipoteza do dalszego sprawdzenia: przewaga bloków z gładkimi podwarstwami przy małym budżecie zależy od zadania (gładka granica decyzyjna 2D vs obrazy).

- **Obserwacja nieplanowana (diagnostyka biegów) [obliczone poza `final.py`]:** w części biegów kryterium wczesnego zatrzymania warstwy 2 wybrało pierwszą epokę (indeks 0) (np. `K-R` MNIST 36/120, `K-G` F-MNIST 45/120, `K-GLT` 24/19; zliczenia z `C/runs/*.json`, pole `best_epoch.layer2`). Do zbadania, czy warstwa 2 w tych ramionach wnosi coś do klasyfikacji (por. E6).

## 6. Ograniczenia

**Braki względem protokołu i szablonu (do uzupełnienia):** E4 (metryki drugorzędne), E5 (krajobraz strojenia – w §7.7 „obowiązkowo”), E8 (udział grup BANK), Rysunek 2 (E9). Zamrożony `final.py` ich nie liczy.

Pełna lista: PROTOKOL_v5 §13 (1–14) oraz odstępstwa z `ODSTEPSTWA.md` (4 pozycje). Szczególnie istotne dla interpretacji:
- §13.11: szeroki próg S na F-MNIST (1.20/1.35 p.p.) – równoważność w H2 na F-MNIST jest równoważnością w szerokim pasie.
- §13.12: w E9 konfiguracje przeniesione z w = 250 bez ponownego strojenia; mniejsze sieci mogły być dostrojone gorzej i nierówno między ramionami.
- E2 i E9 mają N = 10; E4, E5, E8 nie zostały wyliczone przez zamrożony skrypt analizy.
- Wnioski dotyczą jednej architektury (2 warstwy FF, w = 250), jednej funkcji dobroci (Σa²) i dwóch zbiorów danych.
