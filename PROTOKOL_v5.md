# Protokół badania: heterogeniczność i kolejność nieliniowości w sieciach Forward-Forward przy stałym budżecie parametrów

**Wersja:** v5 (finalna do zamrożenia; zamrożenie po §9 i §10)
**Data:** 23.10.2026
**Autor:** Sławomir Majcher
**Katalog:** `/vectorlegis_ssd_pool/uczelnia/eksperyment_ff_od_zera/`

**Dokumenty powiązane:**
- `00_AUDYT_BLEDOW.md`: błędy poprzednich eksperymentów (S*, L*, K*, G*)
- `recenzje/R1.md` … `R3.md` + `odpowiedzi/O1.md` … `O3.md`: rundy 1–3
- `recenzje/R4_niezalezna.md` + `odpowiedzi/O4.md`: niezależna recenzja v4
- `recenzje/R5_weryfikacja.md`: weryfikacja v5 (werdykt: gotowy do zamrożenia po drobnych poprawkach, wprowadzonych w §12.5)
- `DECYZJE_AUTORA.md`: decyzje D1 (Vast.ai, limit 38 USD), D2 (kontrola wyjścia tanh), D3 (analizy „trasy”)

Ten protokół zastępuje protokoły z `eksperyment_sca_ff`, `eksperyment_laminar_ff` i `eksperyment_kontrola_wymiaru_ff`. Ich wyniki służą wyłącznie jako dane pilotowe do analizy mocy i szacunku obciążenia.

---

## 1. Motywacja, pytanie badawcze i zakres

### Motywacja (D3)

Sieć neuronową można opisać jako ciąg deformacji przestrzeni wejściowej albo, równoważnie z punktu widzenia obserwatora, jako nawigację punktów w tej przestrzeni (transformacja aktywna vs pasywna).

Sieć ReLU realizuje deformację kawałkami liniową: jedzie po odcinkach prostych i skręca na ostrych zakrętach. Taka sieć jest uniwersalnym aproksymatorem, ale do odwzorowań gładkich potrzebuje wielu kawałków liniowych (Montúfar i in., 2014; Yarotsky, 2017).

Hipoteza robocza: bogatszy zestaw nieliniowości (gładkie, nasycające, odwracalne) pozwala uzyskać potrzebną deformację mniejszym kosztem parametrów.

W Forward-Forward (FF) nieliniowość wyjściowa wchodzi bezpośrednio w funkcję celu: dobroć wynosi Σa². Wybór nieliniowości zmienia więc także geometrię samego uczenia, nie tylko kształt neuronu.

Analogia z warstwami kory mózgowej pojawiła się w poprzednich eksperymentach po fakcie. Nie jest motywacją ani przedmiotem tego badania.

### Pytania konfirmacyjne

Dotyczą sieci MLP uczonych FF przy równym budżecie parametrów na warstwę FF, równej głębokości i równym budżecie strojenia:

1. **H1.** Czy blok trzech podwarstw LeakyReLU → GELU → tanh („blok LGT”) daje inną dokładność niż **każdy** blok referencyjny o tej samej topologii? Grupa „bloków referencyjnych” obejmuje:
   - bloki jednorodne: ReLU, LeakyReLU, GELU, tanh;
   - bloki L→L→T i G→G→T (jednorodne poza ostatnią podwarstwą tanh; D2).
2. **H2.** Czy blok LGT daje inną dokładność niż **każda** inna kolejność tych samych trzech nieliniowości?
3. **H3.** Czy w płaskiej sieci FF warstwa z równoległym bankiem LeakyReLU | GELU | tanh (dobroć liczona jako Σa² po wszystkich jednostkach) daje inną dokładność niż warstwa z **każdą** z czterech pojedynczych nieliniowości?

### Zakres twierdzeń

- H1 i H2 dotyczą tego konkretnego bloku (L→G→T, bez normalizacji wewnątrz bloku), a nie „różnorodności nieliniowości” w ogóle.
- H3 dotyczy banku z dobrocią Σa² po wszystkich jednostkach.
- Przypisywanie mechanizmu (np. „różnorodność”, „kolejność”, „ograniczona dobroć”) jest dozwolone wyłącznie w sekcji eksploracyjnej raportu.
- Hipoteza „trasy” z motywacji jest badana wyłącznie eksploracyjnie (E9, E10).

### Czego badanie nie rozstrzyga

- zachowania na danych innych niż MNIST i Fashion-MNIST,
- architektur splotowych,
- innych schematów FF.

### Status

- H1–H3: konfirmacyjne (§2, §7.3–7.4).
- Pozostałe analizy: eksploracyjne (§7.7).

---

## 2. Hipotezy konfirmacyjne

Dla pary (ramię badane B, komparator K) i zbioru danych: Δ_K = Acc_B − Acc_K, w punktach procentowych, na zbiorze testowym, parowane po ziarnie (§4.3).

| ID | Ramię badane | Grupa komparatorów (każdy osobno) | m | Przewidywanie autora |
|---|---|---|---|---|
| **H1** | `K-LGT` | `K-R`, `K-L`, `K-G`, `K-T`, `K-LLT`, `K-GGT` | 6 | przewaga |
| **H2** | `K-LGT` | `K-LTG`, `K-GLT`, `K-GTL`, `K-TLG`, `K-TGL` | 5 | przewaga |
| **H3** | `P-BANK` | `P-R`, `P-L`, `P-G`, `P-T` | 4 | przewaga |

Werdykt jest definiowany względem **całej grupy** komparatorów (testy IUT i test sumy, §7.3). Nie ma osobnego wyboru „najlepszego” komparatora.

Testy są dwustronne, a każdy wynik ma z góry zapisaną interpretację (§7.4). Wynik przeciwny przewidywaniu jest raportowany na równi ze zgodnym.

---

## 3. Ramiona

### 3.1. Topologie (szerokość główna w = 250)

**Płaska (P).** Dwie warstwy FF, każda to Linear + nieliniowość.
- Warstwa 1: 784 → 250 (196 250 parametrów).
- Warstwa 2: 250 → 250 (62 750 parametrów).
- **Razem 259 000.**

**Blokowa (K).** Dwie warstwy FF, każda to blok trzech podwarstw. Każda podwarstwa to Linear + nieliniowość, więc w bloku nie ma złożeń liniowo-liniowych.
- Wewnątrz bloku gradient straty FF przechodzi przez wszystkie trzy podwarstwy.
- Między warstwami FF gradient nie przechodzi (`detach`).
- Blok 1: 784 → 163 → 163 → 250 (195 687 parametrów).
- Blok 2: 250 → 103 → 103 → 250 (62 565 parametrów).
- **Razem 258 252**, czyli −0.29% względem P; także każda warstwa FF osobno ma −0.29%.

### 3.2. Nieliniowości

| Symbol | Funkcja |
|---|---|
| R | ReLU |
| L | LeakyReLU, nachylenie ujemne 0.5 |
| G | GELU, `approximate='none'` |
| T | tanh |
| BANK | ciągłe grupy jednostek L \| G \| T o licznościach ⌈n/3⌉, ⌈(n−1)/3⌉, ⌊n/3⌋ |

Przykładowe podziały BANK: 84/83/83 dla n = 250, 55/54/54 dla n = 163, 35/34/34 dla n = 103.

Wyjściem warstwy FF jest wyjście ostatniej podwarstwy. Z niego liczona jest dobroć.

### 3.3. Ramiona badania głównego (18 ramion FF)

| Ramię | Topologia | Podwarstwa 1→2→3 | Wyjście | Rola |
|---|---|---|---|---|
| `P-R`, `P-L`, `P-G`, `P-T` | P | jedna nieliniowość | R/L/G/T | komparatory H3 |
| `P-BANK` | P | BANK | mieszane | badane H3 |
| `K-R`, `K-L`, `K-G`, `K-T` | K | X→X→X | X | komparatory H1 |
| `K-LLT`, `K-GGT` | K | L→L→T, G→G→T | T | komparatory H1 (D2) |
| `K-LGT` | K | L→G→T | T | badane H1, H2 |
| `K-LTG`, `K-GLT`, `K-GTL`, `K-TLG`, `K-TGL` | K | permutacje | G/T/L/G/L | komparatory H2 |
| `K-BANK` | K | BANK→BANK→BANK | mieszane | eksploracyjne (E3) |

**Ramiona BP** (eksploracyjne, E2): `BP-P-R` (261 510 parametrów) i `BP-K-LGT` (260 762 parametrów).
- Topologie P i K z odczytem 250 → 10.
- Uczenie end-to-end entropią krzyżową, na obrazach bez nałożonej etykiety.

`P-R` to „FF w stylu Hintona” w małej skali, a nie odtworzenie Hintona (2022). Tamten wynik (1.36% błędu) dotyczy sieci 4 × 2000 ReLU, 60 epok, innych negatywów i inferencji bez pierwszej warstwy.

### 3.4. Ramiona analiz „trasy” (eksploracyjne)

#### E9: przegląd szerokości (MNIST, F-MNIST)

- **Ramiona:** `P-R`, `P-BANK`, `K-R`, `K-LGT`, `K-LLT`, `K-GGT`.
- **Szerokości:** w ∈ {32, 64, 128}. Punkt w = 250 pochodzi z badania głównego.
- **Wymiary K** dobrane tak, by każda warstwa FF miała budżet najbliższy odpowiadającej warstwie P:

| w | Blok 1 | Blok 2 | Parametry P | Parametry K | Różnica |
|---|---|---|---|---|---|
| 32 | 784 → 30 → 30 → 32 | 32 → 13 → 13 → 32 | 26 176 | 26 531 | +1.36% |
| 64 | 784 → 55 → 55 → 64 | 64 → 26 → 26 → 64 | 54 400 | 53 959 | −0.81% |
| 128 | 784 → 99 → 99 → 128 | 128 → 53 → 53 → 128 | 116 992 | 117 026 | +0.03% |

#### E10: spirala 2D (§4.4)

- **Ramiona:** `P-R`, `P-T`, `P-BANK`, `K-R`, `K-LGT`, `K-GGT` oraz `BP-P-R` jako odniesienie.
- **Budżety** wyznaczone szerokością P: w ∈ {8, 16, 32, 64}.
- **Topologia K w E10:** jednolita szerokość u we wszystkich podwarstwach, dobrana tak, by łączna liczba parametrów była najbliższa P. Wartości: u = 4, 7, 15, 29.
- Dla wejścia 4-wymiarowego dopasowanie warstwa po warstwie dawałoby podwarstwy o szerokości 2–3, stąd dopasowanie łączne. Liczby parametrów raportowane są przy każdym punkcie.

---

## 4. Dane

### 4.1. Zbiory główne

- **MNIST** (LeCun i in., 1998) i **Fashion-MNIST** (Xiao i in., 2017): po 60 000 obrazów treningowych i 10 000 testowych.
- Przetwarzanie: piksele /255, bez augmentacji, wektor 784.
- Pliki źródłowe (gz) są kopiowane na instancję (`rsync`) razem z kodem. Ich hashe SHA-256 są w `dane/SHA256SUMS_DANE` (część `SHA256SUMS`, §10); runner weryfikuje je przy każdym wczytaniu, a skrypt wdrożenia dodatkowo po skopiowaniu (por. R4-10; zmiana przed zamrożeniem, §12.6).
- **QMNIST** (Yadav i Bottou, 2019): dodatkowe 50 000 cyfr testowych (`what='test50k'`), używane wyłącznie w analizie wrażliwości §7.8(d). Hash także w `SHA256SUMS`.

### 4.2. Podziały

- **Zbiór testowy (10 000)** jest używany wyłącznie w ewaluacji końcowej (§5.6). Obejmuje ona Etap C, E2 i E9. Runner w trybach treningowych nie ma ścieżki do zbioru testowego (test 9.1.5).
- **Podział treningowy** jest losowany per ziarno: 50 000 treningowych / 10 000 walidacyjnych.
- **Wczesne zatrzymanie** korzysta ze stałego (per ziarno) podzbioru 5 000 przykładów walidacyjnych.
- **Wybór konfiguracji w Etapie B** korzysta z pełnych 10 000 przykładów walidacyjnych.

### 4.3. Ziarna, strumienie, parowanie

**Strumienie losowe.** Każdy strumień to osobny generator `torch.Generator` na CPU. Wylosowane indeksy i etykiety są przenoszone na GPU. Ziarno generatora to pierwsze 63 bity SHA-256 z tekstowej reprezentacji krotki:

| Strumień | Krotka | Zastosowanie |
|---|---|---|
| `split` | (s, 1) | podział danych i podzbiór 5k |
| `order` | (s, 2, warstwa, epoka) | kolejność przykładów |
| `neg` | (s, 3, warstwa, epoka) | negatywy |
| `init` | (s, 4) | inicjalizacja wag |

Strumienie `split`, `order` i `neg` są identyczne dla wszystkich ramion przy tym samym ziarnie. Nie zależą od chwili zatrzymania warstwy 1.

**Pule ziaren** (rozłączne między sobą i z pilotami):

| Pula | Ziarna |
|---|---|
| Mini-pilot (§9.3) | 40001–40010 |
| Etap A | 30001 |
| Etap B | 20001–20003 |
| Etap C | 10001 … 10000+N, gdzie N z §7.5 |
| E2 i E9 | 10001–10010 (parowanie z badaniem głównym) |
| E10 | 50000 (strojenie), 50001–50010 (ewaluacja) |

### 4.4. Spirala 2D (E10)

- **Dwie klasy**, dwa przeplecione ramiona spirali o 1.5 obrotu:
  - x = t · (cos 3πt, sin 3πt), t ∈ [0.1, 1];
  - druga klasa obrócona o π.
- **Szum:** gaussowski, σ = 0.04.
- **Liczebność:** po 500 punktów na klasę, generowanych ze stałym ziarnem 0.
- **Podział:** 600 treningowych / 200 walidacyjnych / 200 testowych.
- **Wejście FF:** (x₁, x₂) konkatenowane z one-hot etykiety (2 wymiary), razem 4 wymiary. W 2D nie ma pikseli do nadpisania.
- **Wejście BP:** (x₁, x₂).

---

## 5. Procedura

### 5.1. Forward-Forward

- **Etykieta:** piksele 0–9 są zerowane, a piksel o indeksie y ustawiany na 1.0, po skalowaniu obrazu. Wyjątek: w E10 etykieta jest konkatenowana (§4.4).
- **Negatywy:** etykieta y' ~ U({0..9} \ {y}), losowana w każdym batchu ze strumienia `neg`.
- **Dobroć:** G = Σⱼ aⱼ².
- **Strata:** L = mean softplus(θ − G_pos) + mean softplus(G_neg − θ).
- **Próg:** θ = c · w, gdzie w to szerokość wyjścia warstwy FF; c jest wspólne dla obu warstw FF.
- **Normalizacja między warstwami:** wejście warstwy 2 to a / (‖a‖₂ + 1e-8).
- **Uczenie zachłanne:** najpierw warstwa 1 do zatrzymania, potem jest zamrażana i uczona jest warstwa 2.
- **Buforowanie:** wyjścia zamrożonej warstwy 1 dla 10 wariantów etykiety są liczone raz, w stałych porcjach po 1 000 przykładów (por. 9.1.9).
- **Optymalizacja:** AdamW, batch 128, `drop_last=True`, stały lr, przycinanie normy gradientu do 10.
- **Klasyfikacja:** ŷ = argmax_c Σₗ Gₗ(x z etykietą c).

### 5.2. Epoki i zatrzymanie

- **Wczesne zatrzymanie:** cierpliwość 5 epok, mierzona na dokładności podzbioru 5k. Dla warstwy 1 dokładność liczona jest z G₁, dla warstwy 2 z G₁ + G₂. Przywracane są wagi z najlepszej epoki.
- **Limit epok:** E_max = max(30, ⌈1.5 × P90 epoki najlepszej w mini-pilocie⌉), nie więcej niż 80. P90 jest liczony po obu warstwach i wszystkich ramionach mini-pilota.
- **Reguła limitu aktywnego** (por. R4-07). Sprawdzana jest po Etapie A (zmiana obowiązuje od Etapu B) i ponownie po Etapie B (zmiana obowiązuje od Etapu C):
  - jeśli dla któregokolwiek ramienia więcej niż 25% biegów etapu ma najlepszą epokę ≥ E_max − 2 w którejkolwiek warstwie,
  - to E_max dla **wszystkich** ramion wzrasta o 50%.
  - Reguła może zadziałać najwyżej dwa razy.
  - Decyzje są zapisywane w `ETAP_A_WYBOR.json` i `ETAP_B_WYBOR.json`.

### 5.3. Etap A: strojenie z równym budżetem

- **Przestrzeń:**
  - lr ∈ [1e-4, 1e-2] (skala logarytmiczna),
  - wd ∈ [1e-6, 1e-2] (log),
  - c ∈ [0.01, 1] (log).
- **Punkty:** 16 punktów scrambled Sobol (ziarno 0), identycznych dla 18 ramion FF.
- **Ramiona BP** dostają te same 16 par (lr, wd).
- Na każdy punkt przypada 1 bieg na ziarnie 30001. Kryterium: dokładność na pełnej walidacji.
- Strojenie przeprowadza się osobno dla każdego zbioru danych.

### 5.4. Etap B: wybór konfiguracji

- **Kandydaci:** dla każdego ramienia i zbioru 3 najlepsze punkty z Etapu A, każdy oceniany na 3 ziarnach (20001–20003).
- **Konfiguracja ramienia:** punkt o najwyższej średniej dokładności walidacyjnej. Przy remisie wygrywa niższe c.
- **Reguła brzegowa:**
  1. Oś „brzegowa” to oś, na której konfiguracja któregokolwiek ramienia (w danym zbiorze) leży w skrajnych 10% zakresu w skali log.
  2. Brane są najwyżej 2 osie brzegowe. Jeśli są 3, wybierane są dwie z największą liczbą ramion na brzegu; remis rozstrzyga kolejność lr, wd, c.
  3. Dla każdej osi brzegowej **każde** ramię dostaje 2 punkty: własna konfiguracja przesunięta o ±25% szerokości osi w skali log.
  4. c jest obcinane do 1. Punkt, który po obcięciu powtarza już oceniony punkt, jest pomijany.
  5. Nowe punkty są oceniane na tych samych 3 ziarnach, po czym wybór konfiguracji jest powtarzany spośród wszystkich ocenionych punktów.
  6. Reguła jest stosowana jednokrotnie, a budżet zakłada jej uruchomienie.
- **Wynik:** `ETAP_B_WYBOR.json` zawiera konfiguracje, E_max i decyzje. Jest commitowany i znakowany czasem (§10) **przed** Etapem C.
- **Konfiguracje E2 i E9:**
  - E2 (BP) używa konfiguracji BP wybranej w Etapie B według tej samej procedury co dla FF: 3 najlepsze punkty (lr, wd) z Etapu A × 3 ziarna, reguła brzegowa na osiach lr i wd (por. R5-02).
  - E9 używa konfiguracji ramienia z w = 250, bez ponownego strojenia; θ = c · w. Jest to ograniczenie E9 (§13).

### 5.5. Etap C: trening konfirmacyjny

- **Zakres:** 18 ramion × N ziaren × 2 zbiory. Równolegle trenowane są modele E2 (2 ramiona × 10 ziaren × 2 zbiory) i E9 (6 ramion × 3 szerokości × 10 ziaren × 2 zbiory).
- **Zapisywane per bieg:**
  - wagi z najlepszej epoki,
  - krzywe uczenia,
  - metryki walidacyjne.
- Metryki walidacyjne Etapu C, E2 i E9 trafiają do katalogu `zaslepione/`. Autor nie otwiera go przed krokiem 2 z §5.6 (por. R4-06). Wczesne zatrzymanie działa wewnątrz runnera i tego katalogu nie potrzebuje.
- **Ewaluacji testowej w tym etapie nie ma.**

### 5.6. Ewaluacja końcowa (zaślepiona)

1. Po zakończeniu **wszystkich** biegów Etapu C, E2 i E9 commitowany jest plik z hashami SHA-256 wszystkich checkpointów.
2. Jedno polecenie ocenia wszystkie checkpointy na zbiorze testowym (i na QMNIST-50k dla modeli MNIST) i zapisuje predykcje per przykład.
3. Uruchamiany jest skrypt analizy zamrożony w §10.
4. **Powtórki przed krokiem 2** są dopuszczalne wyłącznie po **błędzie kodu**:
   - **Definicja** (por. R4-06): błąd kodu to niezgodność implementacji z §3, §5 lub §5.7, wykazana testem, który nie przechodzi na kodzie pierwotnym i przechodzi na poprawionym. Test jest commitowany razem z diffem. Zmiana, której nie da się wykazać w ten sposób, nie uzasadnia powtórki.
   - **Zakres powtórki:** wszystkie biegi etapu, które przechodzą przez zmieniony kod. Wyznacza je automatycznie skrypt pokrycia (funkcje z diffu → biegi). W razie wątpliwości powtarzane są wszystkie biegi etapu.
   - **Zapis:** diff, test, lista biegów i powód trafiają do `ODSTEPSTWA.md` i są commitowane **przed** ponownym uruchomieniem.
   - **Biegi pierwotne:** ich checkpointy są zachowywane i oceniane w kroku 2, a wyniki raportowane w §7.8(c).
5. Powtórki po kroku 2 są dopuszczalne tylko jako analiza wrażliwości. Wynikiem głównym pozostaje wynik pierwotny.
6. **Przerwanie badania** (np. brak środków po wstrzymaniu, §11.3): ewaluacja testowa częściowo ukończonego Etapu C jest zabroniona. Raport opisuje badanie jako nieukończone i podaje jedynie wyniki Etapów A–B oraz mini-pilota.

### 5.7. Specyfikacja implementacyjna

| Element | Wartość |
|---|---|
| Inicjalizacja | domyślna PyTorch `nn.Linear` (wagi Kaiming-uniform a = √5; bias U(±1/√fan_in)), jednakowa dla wszystkich ramion |
| Bias | tak, w każdej warstwie liniowej |
| dtype | float32 |
| AdamW | betas (0.9, 0.999), eps 1e-8 |
| Eps normalizacji | 1e-8 |
| BANK | podział według §3.2; dobroć to suma po wszystkich jednostkach |
| BP | entropia krzyżowa, AdamW, batch 128, przycinanie 10, wczesne zatrzymanie jak w §5.2 (dokładność walidacyjna), przywracanie najlepszych wag |
| Sprzęt | GPU wynajęte w Vast.ai (D1). Cały zbiór danych w pamięci GPU. k biegów równolegle na karcie, każdy w osobnym procesie; k ustalane w §9.3 i stałe |
| Determinizm | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Wątki CPU | `torch.set_num_threads(1)` na proces, `set_num_interop_threads(1)`; k ≤ liczba vCPU instancji |
| Środowisko | oficjalny obraz `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` na całe badanie; dane i kod kopiowane z weryfikacją hashy (§4.1); wersje Pythona, PyTorch, CUDA, GPU, sterownik hosta i commit kodu zapisywane w JSON każdego biegu |
| E10 | CPU lokalnie (sieci < 5 000 parametrów); full-batch AdamW; E_max = 5 000 epok na warstwę, cierpliwość 1 000 (zmiana przed zamrożeniem, §12.6); strojenie: 8 punktów Sobola z przestrzeni §5.3, identycznych dla wszystkich ramion i budżetów, na walidacji spirali przy ziarnie 50000 (spoza ziaren ewaluacyjnych); punkt wybierany osobno dla każdej pary (ramię, budżet) (por. R5-03) |

Inicjalizacja domyślna nie jest optymalna dla wszystkich nieliniowości. Wyniki są warunkowe względem niej (§13).

---

## 6. Metryki

### 6.1. Główna

Dokładność testowa w procentach (§5.6), z zapisanymi predykcjami per przykład.

### 6.2. Drugorzędne (opisowe, §7.7)

Liczone na pozytywach walidacji, per warstwa FF:
- **Jednostki nieaktywne:** sd(aⱼ) < 0.01 · P90ₖ sd(aₖ), wraz z pełnym rozkładem sd.
- **Participation ratio:** (Σλ)² / Σλ², liczone z wartości własnych kowariancji wyjść.
- **Udział grup BANK:** udział L, G i T w G_pos, G_neg oraz w normie gradientu (E8).
- **Przebieg uczenia:**
  - krzywe straty i dokładności walidacyjnej,
  - epoka najlepsza,
  - odsetek biegów z aktywnym limitem epok,
  - odsetek biegów z NaN.
- **Czas i koszt:** opisowo.

---

## 7. Analiza statystyczna

### 7.1. Estymand

Estymandem jest różnica dokładności **na oficjalnym zbiorze testowym**, uśredniona po losowości treningu. Losowość treningu obejmuje inicjalizację, kolejność przykładów, negatywy i podział 50k/10k.

Wnioski są warunkowe względem tego zbioru testowego. Analizy §7.8(a) i §7.8(d) oceniają, czy wnioski wykraczają poza niego.

### 7.2. Test elementarny i kontrola błędu

- dᵢ = Acc_B(sᵢ) − Acc_K(sᵢ). Do tego przedział ufności t-Studenta dla średniej d.
- Brak testu normalności i brak warunkowego wyboru testu.
- α = 0.05 z korektą Bonferroniego na 3 hipotezy, czyli **α' = 0.0167**.

**Kontrola błędu** (por. R3, R4-11):
- Prawdopodobieństwo fałszywego werdyktu jest kontrolowane **osobno dla każdego typu twierdzenia**:
  - klasa P (przewaga): ≤ α'/2,
  - klasa G (gorszość): ≤ α'/2,
  - klasa R (równoważność): ≤ α'.
- Łączne prawdopodobieństwo fałszywego werdyktu dowolnego typu w jednej hipotezie może sięgnąć 1.5α'.
- W całym badaniu (3 hipotezy) daje to ≤ 0.05 dla każdego typu twierdzenia osobno.

### 7.3. Werdykt hipotezy na jednym zbiorze

**Oznaczenia:**
- CI_K: dwustronny przedział (1 − α') dla Δ_K.
- CI2_K: przedział (1 − 2α') dla Δ_K, używany w teście TOST.
- S: SESOI, czyli najmniejszy efekt uznawany za praktycznie istotny (§7.5).

**Warunki elementarne:**

| Warunek | Definicja | Typ |
|---|---|---|
| **sup(S)** | dla każdego K: dolna granica CI_K > S | IUT |
| **sup(0)** | dla każdego K: dolna granica CI_K > 0 | IUT |
| **eq** | dla każdego K: CI2_K ⊂ (−S, S) | IUT (TOST) |
| **inf(S)** | jednostronne p_K dla H₀: Δ_K ≥ −S; procedura Holma na m wartościach p, poziom α'/2; odrzucono ≥ 1 hipotezę | test sumy |
| **inf(0)** | jak inf(S), dla H₀: Δ_K ≥ 0 | test sumy |

**Kategorie** (pierwsza spełniona kategoria jest werdyktem):

| # | Kategoria | Warunek | Klasa |
|---|---|---|---|
| 1 | GORSZOŚĆ | inf(S) | G |
| 2 | PRZEWAGA | sup(S) | P |
| 3 | RÓWNOWAŻNOŚĆ+ | eq ∧ (sup(0) ∨ inf(0)) | R |
| 4 | GORSZOŚĆ<S | inf(0) | G |
| 5 | PRZEWAGA<S | sup(0) | P |
| 6 | RÓWNOWAŻNOŚĆ | eq | R |
| 7 | NIEROZSTRZYGNIĘTY | – | N |

**Własności kategorii:**
- sup(S) i inf(S) wykluczają eq.
- sup(·) i inf(·) nie mogą zachodzić jednocześnie.
- Kategorie 4–5 to różnice istotne statystycznie bez wykazanej równoważności. Ich wielkość praktyczna pozostaje niepewna.

### 7.4. Łączenie zbiorów i interpretacja

**Łączenie** (IUT po zbiorach, Berger i Hsu, 1996):

| Sytuacja | Werdykt łączny |
|---|---|
| Ta sama klasa na obu zbiorach | Ta klasa. Kategoria łączna jest słabszą z dwóch: P → PRZEWAGA tylko przy PRZEWADZE na obu zbiorach; analogicznie G; R → RÓWNOWAŻNOŚĆ+, jeśli wystąpi choć na jednym zbiorze |
| P na jednym zbiorze, G na drugim | NIESPÓJNY-KIERUNEK (klasa N) |
| P na jednym zbiorze, R+ o dodatnim kierunku na drugim (lub G i R+ o ujemnym kierunku) | NIESPÓJNY-ZGODNY-KIERUNEK (klasa N), por. R4-09 |
| Pozostałe kombinacje | NIESPÓJNY-CZĘŚCIOWY (klasa N) |

NIESPÓJNY-ZGODNY-KIERUNEK ma w szablonie osobne zdanie (por. R5-05): „na zbiorze X: [kategoria zbiorowa]; na zbiorze Y: różnica istotna statystycznie w tym samym kierunku, praktycznie pomijalna (< S, równoważność wykazana)”.

**Obowiązkowe dopiski:**
- Przy kategoriach „<S”: „różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane”.
- Przy RÓWNOWAŻNOŚCI+: „wykryto różnicę istotną statystycznie, ale praktycznie pomijalną (< S)”, z kierunkiem na **każdym** zbiorze.
- Przy klasie G w H1 lub H2: lista komparatorów, które są istotnie lepsze od LGT, wraz z ich nieliniowością wyjściową (por. R4-03).
- Przy każdej komórce z klasą P lub G: która wersja zaszła (pełna czy „<S”).

**Macierz H1 × H2.** Zdania opisowe, bez przypisywania mechanizmu (por. R4-03). „Bloki referencyjne” oznaczają całą grupę komparatorów H1: 4 bloki jednorodne oraz `K-LLT` i `K-GGT` (por. R5-01).

| H1 \ H2 | P | R | G | N |
|---|---|---|---|---|
| **P** | LGT lepszy od każdego bloku referencyjnego (także `K-T`, `K-LLT`, `K-GGT`) i od każdej innej kolejności. | LGT lepszy od każdego bloku referencyjnego; równoważny każdej innej kolejności (±S). | LGT lepszy od każdego bloku referencyjnego; co najmniej jedna inna kolejność lepsza od LGT. | LGT lepszy od każdego bloku referencyjnego; porównanie z innymi kolejnościami nierozstrzygnięte. |
| **R** | LGT równoważny blokom referencyjnym (±S) i lepszy od każdej innej kolejności; porównanie innych kolejności z blokami referencyjnymi jest eksploracyjne (E3, E7). | LGT równoważny każdemu blokowi referencyjnemu i każdej innej kolejności (±S). | LGT równoważny blokom referencyjnym; co najmniej jedna inna kolejność lepsza od LGT. | LGT równoważny blokom referencyjnym; porównanie z innymi kolejnościami nierozstrzygnięte. |
| **G** | LGT gorszy od co najmniej jednego bloku referencyjnego i lepszy od każdej innej kolejności. | LGT gorszy od co najmniej jednego bloku referencyjnego; pozostałe kolejności równoważne LGT (±S). | LGT gorszy od co najmniej jednego bloku referencyjnego i od co najmniej jednej innej kolejności. | LGT gorszy od co najmniej jednego bloku referencyjnego; porównanie z innymi kolejnościami nierozstrzygnięte. |
| **N** | Porównanie z blokami referencyjnymi nierozstrzygnięte; LGT lepszy od każdej innej kolejności. | Porównanie z blokami referencyjnymi nierozstrzygnięte; inne kolejności równoważne LGT. | Porównanie z blokami referencyjnymi nierozstrzygnięte; co najmniej jedna inna kolejność lepsza od LGT. | Badanie nie rozstrzyga H1 ani H2. |

Przy H1 = P dopuszczalne jest zdanie wynikające wprost z testu: „przewaga LGT nie sprowadza się do obecności tanh na wyjściu bloku referencyjnego” (komparatory `K-T`, `K-LLT`, `K-GGT`).

**H3:**

| Klasa | Zdanie |
|---|---|
| P | Bank (Σa² po wszystkich jednostkach) lepszy od każdej pojedynczej nieliniowości w topologii płaskiej. |
| R | Bank równoważny każdej pojedynczej nieliniowości (±S). |
| G | Co najmniej jedna pojedyncza nieliniowość lepsza od banku. |
| N | Nierozstrzygnięty. |

Każde zdanie jest ograniczone zakresem z §1 i §13.

### 7.5. SESOI, liczebność próby i moc

**SESOI.** Wartość S wynosi 10% błędu walidacyjnego z mini-pilota (§9.3), osobno dla każdego zbioru, zaokrąglone w górę do 0.05 p.p.:
- H1 i H2: S = 0.10 × błąd `K-LGT`,
- H3: S = 0.10 × błąd `P-R`.

Konfiguracja mini-pilota nie jest strojona, więc S może być nieco szersze niż 10% błędu modelu strojonego.

**Parametry symulacji mocy** (najgorszy przypadek):
- **m = 6** (H1).
- **S** = min(S dla H1/H2, S dla H3) danego zbioru.
- **σ_d** (dla każdego zbioru, por. R4-01) = max(1.29 × max SD różnic, 0.5 · S), gdzie maksimum SD liczone jest po kontrastach wewnątrzgrupowych z mini-pilota:
  - `K-LGT − K-G`,
  - `K-LGT − K-T`,
  - `K-LGT − K-GGT`,
  - `K-LGT − K-GLT`,
  - `P-BANK − P-R`,
  - `P-BANK − P-T`.
  
  Czynnik 1.29 to górna granica 80% przedziału ufności SD przy 9 stopniach swobody.
- **ρ** (korelacja różnic w grupie) = min(ρ̂, 0.5), nie mniej niż 0. ρ̂ to średnia korelacja par różnic `K-LGT − K-x` w mini-pilocie. Niższa korelacja zmniejsza moc testów IUT, więc ograniczenie z góry działa zachowawczo.
- **Moc całego potoku** liczona jest symulacją §7.3–7.4 dla obu zbiorów.

**Cele mocy:**
- (i) P(klasa P) ≥ 0.80, gdy Δ = S względem jednego komparatora i 2S względem pozostałych;
- (ii) P(klasa R) ≥ 0.80, gdy Δ = 0 względem wszystkich komparatorów.

Kategoria PRZEWAGA (> S) jest drugorzędna. Jej moc dla Δ = 2S/3S jest tylko raportowana.

**Reguła N.** N to najmniejsza wartość z {30, 40, 50, 60} spełniająca cele (i) i (ii). Jeśli żadna jej nie spełnia, N = 60, a uzyskane moce zostają wpisane do protokołu.

**Wpisywane przed zamrożeniem:**
- N, S i σ_d per zbiór,
- ρ̂,
- moce przy σ_d i przy 1.47 × max SD,
- składowa wariancji z próbkowania przykładów, oszacowana na walidacji 10k w mini-pilocie.

**Brak analiz pośrednich.** Monitor postępu pokazuje wyłącznie liczbę ukończonych biegów, błędy i NaN.

### 7.6. NaN i niestabilność

- **Zatrzymanie:** bieg z NaN w stracie zatrzymuje uczenie danej warstwy i zostaje w analizie.
- **NaN w warstwie 2** bez żadnej epoki wolnej od NaN: klasyfikacja wyłącznie przez argmax G₁ (por. R4-08).
- **NaN w warstwie 1** przed pierwszą ukończoną epoką: wyjście losowe ze strumienia `init` (≈ 10%).
- **Wilcoxon przy NaN:** jeśli w hipotezie i zbiorze wystąpi choć jeden bieg z NaN, werdykt liczony jest dodatkowo testem Wilcoxona (te same progi, przedział Hodgesa–Lehmanna). Jeśli kategorie się różnią, werdykt brzmi NIEROZSTRZYGNIĘTY (niestabilność).

### 7.7. Analizy eksploracyjne

Analizy eksploracyjne nie dają wniosków konfirmacyjnych, nie mają korekty na wielokrotne porównania i zawsze są oznaczane jako eksploracyjne.

| ID | Analiza |
|---|---|
| E1 | Głębokość: `K-x` vs `P-x` dla x ∈ {R, L, G, T, BANK} |
| E2 | Luka do BP (N = 10): `BP-P-R` − `P-R` oraz `BP-K-LGT` − `K-LGT`. BP stroi się w 2D, FF w 3D |
| E3 | Wszystkie kontrasty pojedyncze w grupach H1–H3; `K-BANK`; porównania innych kolejności z blokami referencyjnymi |
| E4 | Metryki drugorzędne (§6.2) |
| E5 | Krajobraz strojenia z Etapu A (obowiązkowo), z zaznaczonym nieosiągalnym zakresem c dla wyjść tanh |
| E6 | Inferencja w wariancie Hintona: tylko G₂ |
| E7 | Wyjście tanh: średnia (LGT, GLT) − średnia (pozostałe 4 permutacje); LGT vs GLT; LGT vs {`K-T`, `K-LLT`, `K-GGT`} |
| E8 | Udział grup BANK w dobroci i gradiencie |
| **E9** | Przegląd szerokości (§3.4). Krzywa dokładności testowej w funkcji liczby parametrów (w ∈ {32, 64, 128, 250}) dla 6 ramion, N = 10, średnie z 95% przedziałami ufności. Przewidywanie autora (zapisane przed badaniem): przewaga ramion heterogenicznych (`K-LGT`, `P-BANK`) nad najlepszym ramieniem referencyjnym tej samej topologii **rośnie, gdy w maleje**. „Najlepsze ramię referencyjne” wybierane jest osobno dla każdego w: najwyższa średnia dokładność walidacyjna (z `zaslepione/`, po kroku 2 §5.6) wśród ramion referencyjnych E9 danej topologii (K: `K-R`, `K-LLT`, `K-GGT`; P: `P-R`). Opisowo: różnica przy w = 32 minus różnica przy w = 250, z przedziałem bootstrap. Zawsze raportowane są także różnice względem **każdego** ramienia E9. Bez testu konfirmacyjnego (por. R5-04) |
| **E10** | Spirala 2D (§3.4, §4.4), 10 ziaren. (a) Dokładność testowa w funkcji liczby parametrów dla każdego ramienia. (b) Najmniejszy budżet, przy którym mediana dokładności testowej ≥ 95%. (c) **Rysunek** (§7.9): mapy decyzyjne |

**Reguła wyboru rysunku E10** (ustalona z góry, chroni przed wybieraniem korzystnych przykładów):
- budżet w = 16;
- dla każdego ramienia ziarno o **dolnej medianie** dokładności walidacyjnej: 5. wartość w porządku rosnącym spośród 10; przy równych dokładnościach niższy numer ziarna (por. R5-03);
- mapa klasy na siatce 300 × 300 punktów w [−1.2, 1.2]², z naniesionymi punktami testowymi.

Dodatkowo, w materiałach uzupełniających: siatka wszystkich ramion × wszystkich budżetów, także dla ziarna medianowego.

### 7.8. Wrażliwość

- **(a) Bootstrap dwuwymiarowy.** Losowanie ziaren i, niezależnie, przykładów testowych; 10 000 replikacji, metoda percentylowa.
  - CI_K: kwantyle (α'/2, 1 − α'/2); CI2_K: kwantyle (α', 1 − α').
  - Dla inf(·): p jednostronne bootstrapu, potem procedura Holma.
  - Bootstrap niezależny w układzie skrzyżowanym jest zachowawczy.
  - Twierdzenia o rozkładzie danych (a nie tylko o zbiorze testowym) są dopuszczalne wyłącznie przy zgodnym werdykcie (a).
- **(b) Test Wilcoxona** zamiast testu t.
- **(c) Powtórki:** wyniki biegów pierwotnych zastąpionych powtórkami przed ewaluacją oraz powtórek po ewaluacji (§5.6).
- **(d) QMNIST-50k** (por. R4-02): werdykty §7.3 dla MNIST na 50 000 dodatkowych cyfr QMNIST.
  - Zbiór testowy MNIST był oglądany w pilotach poprzedzających wybór hipotez. QMNIST-50k nie był używany nigdy.
  - Rozbieżność werdyktów (MNIST-10k vs QMNIST-50k) jest raportowana na pierwszej stronie raportu.

### 7.9. Rysunki do pracy

**Rysunek 1:** mapy decyzyjne E10, według reguły wyboru z §7.7.
- Opis rysunku musi zawierać: ramię, budżet, liczbę parametrów, wybrane ziarno i jego dokładność testową, a także informację, że to ziarno medianowe spośród 10.

**Rysunek 2:** krzywe E9.

Oba rysunki należą do sekcji eksploracyjnej.

---

## 8. Wykonanie (Vast.ai)

**Sprzęt (D1):**
- Typ GPU i obraz Dockera ustalane są przed mini-pilotem i obowiązują we wszystkich etapach.
- Zmiana typu w trakcie etapu jest zabroniona.

**Awaria instancji:**
- Uruchamiana jest nowa instancja tego samego typu.
- Na nowej instancji powtarzany jest test 9.1.6 (wynik do logu, por. R4-05). Brak bitowej zgodności między hostami jest odnotowywany. Nie wpływa na porównania, bo parowanie nie wymaga determinizmu między hostami.
- Biegi przerwane uruchamia się od zera.

**Zmiana typu GPU między etapami:**
- Przed kolejnym etapem powtarzany jest test 9.1.6 i pomiar §9.3.
- Wynik trafia do `ODSTEPSTWA.md`.

**Synchronizacja (por. R4-04):**
- Po każdym ukończonym biegu `rsync` kopiuje JSON i checkpoint na maszynę lokalną, z weryfikacją hashy.
- Bieg jest ukończony dopiero po potwierdzonej kopii lokalnej.
- Przy wznowieniu po awarii biegi bez kopii są uruchamiane ponownie.

**Kolejność biegów:**
- Dla każdego ziarna ramiona i zbiory w losowej kolejności (permutacja ze strumienia ziarna).
- Biegi są atomowe.

**Zawartość JSON per bieg:**
- metryki walidacyjne (w Etapie C, E2 i E9 zapisywane do `zaslepione/`),
- krzywe,
- wersje,
- hash checkpointu,
- **bez dokładności testowej**.

**Monitor:** pokazuje wyłącznie liczbę ukończonych biegów, błędy, NaN i zużyty budżet w USD.

**Instancje:**
- Instancja jest niszczona po każdym etapie, po commicie wyników.
- Każde uruchomienie, zniszczenie i doładowanie wymaga potwierdzenia autora.

**E10:** wykonywane lokalnie na CPU, poza budżetem D1, z tymi samymi zasadami zapisu i ślepoty testu.

---

## 9. Weryfikacja przed zamrożeniem

Kod (runner, analiza, symulacja mocy, skrypt wyliczający parametry z mini-pilota) jest pisany po akceptacji tego protokołu i na jego podstawie.

### 9.1. Testy jednostkowe

1. Liczby parametrów zgodne z §3.1 i §3.4, dla wszystkich szerokości.
2. Każda warstwa liniowa (poza odczytem BP) ma po sobie nieliniowość.
3. Etykieta negatywna jest różna od prawdziwej w 100% przypadków.
4. Uczenie warstwy 2 nie zmienia parametrów warstwy 1.
5. W trybie treningowym zbiór testowy (ani QMNIST) nie może zostać wczytany.
6. Bitowa powtarzalność pary (ramię, ziarno) na docelowym GPU. Test powtarzany na każdej nowej instancji.
7. Strumienie `split`, `order` i `neg` są identyczne dla różnych ramion, także w warstwie 2 przy różnych epokach zatrzymania warstwy 1.
8. Odsetek obrazów z niezerowymi pikselami 0–9. Jeśli przekracza 1%, wpis do §13.
9. **Buforowanie** (por. R4-05). Kryterium tolerancji ustalone z góry: trening z buforem vs trening z przeliczaniem warstwy 1 w tych samych porcjach po 1 000. Wymagane: max |Δa| < 1e-5 na wyjściu warstwy 2 po 1 epoce oraz identyczne predykcje na walidacji 5k.
10. Hashe danych zgodne z `SHA256SUMS`.

### 9.2. Test skryptu analizy na symulacji całego potoku

Dane syntetyczne: 18 ramion, 2 zbiory, N ziaren, korelacje jak w §7.5, z biegami NaN. Każda kontrola na 2 000 symulacji.

| | Prawdziwy stan | Wymaganie |
|---|---|---|
| (a) | Δ = 0 względem jednego komparatora, 2S względem pozostałych | klasa P ≤ α'/2 (z marginesem Monte Carlo) |
| (a′) | Δ = S względem jednego, 3S względem pozostałych | PRZEWAGA ≤ α'/2 |
| (b) | Δ = 0 względem wszystkich | klasa G ≤ α'/2 |
| (c) | scenariusz (i) z §7.5 | moc klasy P zgodna z §7.5 |
| (d) | scenariusz (ii) z §7.5 | moc klasy R zgodna z §7.5 |
| (e) | dowolny | zgodność z implementacją referencyjną na 100 ręcznie skonstruowanych przypadkach (wszystkie kategorie, RÓWNOWAŻNOŚĆ+, wszystkie typy łączenia zbiorów, NaN) |

### 9.3. Mini-pilot pomiarowy

**Ustawienia:**
- Ramiona (por. R4-01): `P-R`, `P-T`, `P-BANK`, `K-LGT`, `K-G`, `K-T`, `K-GGT`, `K-GLT`.
- Konfiguracja stała: lr 1e-3, wd 1e-4, c = 0.25.
- E_max = 60.
- Ziarna 40001–40010, oba zbiory, docelowy GPU (D1).
- **Wyłącznie podział treningowo-walidacyjny; zbiór testowy nie jest używany.**

**Przed mini-pilotem:** pomiar przepustowości na danych losowych z buforem warstwy 1, dla k = 1, 2, 4, … procesów (k ≤ vCPU, w granicach VRAM). Wybierane jest k o największej liczbie biegów na godzinę i od tej chwili pozostaje stałe.

**Produkty mini-pilota:**
1. SD wszystkich kontrastów wewnątrzgrupowych i ρ̂, czyli σ_d i ρ do §7.5.
2. Błędy walidacyjne `K-LGT` i `P-R`, czyli S.
3. Rozkład epok najlepszych, czyli E_max.
4. Czas na bieg i koszt w USD na komplet ramion, czyli budżet §11.
5. Składowa wariancji z próbkowania przykładów (walidacja 10k), raportowana w §7.5.

**Jednokrotność:**
- Mini-pilot wykonywany jest jeden raz.
- Runner mini-pilota i skrypt wyliczający są commitowane **przed** uruchomieniem.
- Surowy `MINIPILOT.json` jest commitowany **przed** wyliczeniami.
- Przerwany bieg uruchamia się od zera.
- Powtórzenie całego mini-pilota jest dopuszczalne tylko po awarii technicznej, z wpisem w `ODSTEPSTWA.md`.

### 9.4. Szablon raportu (results-blind)

`RAPORT_SZABLON.md` zawiera:
- wszystkie tabele z pustymi polami,
- dla H1–H3 wszystkie kategorie i gotowe zdania z §7.4, w tym NIESPÓJNY-ZGODNY-KIERUNEK,
- sekcję eksploracyjną pod nagłówkiem „wyniki eksploracyjne – bez wniosków konfirmacyjnych”, z miejscami na rysunki §7.9.

### 9.5. Zasady języka raportu

- Bez słów „dowodzi”, „bezprecedensowy”, „idealny”, „rozwiązuje”, „deklasuje”.
- Każdy wniosek z zakresem: „MLP, MNIST/Fashion-MNIST, schemat FF z §5, na oficjalnym zbiorze testowym”.
- Wyniki negatywne i nierozstrzygnięte opisane wprost na pierwszej stronie.
- Hipoteza „trasy” (D3) opisywana wyłącznie jako motywacja i przedmiot analiz eksploracyjnych.

---

## 10. Zamrożenie i znaczniki czasu

1. **Zamrożenie.** Następuje po akceptacji protokołu i ukończeniu §9 (w tym wpisaniu wartości z mini-pilota do §5.2, §7.5 i §11). Obejmuje:
   - commit z tagiem `ff-od-zera-prereg`,
   - `SHA256SUMS`: protokół, runner, analiza, symulacja mocy, szablon raportu, `MINIPILOT.json`, pliki danych, hash obrazu Dockera.
2. **Zewnętrzny znacznik czasu** (OSF lub tag w publicznym repozytorium) **przed Etapem A**. Bez niego raport nazywa badanie „zaplanowanym z góry, bez zewnętrznej rejestracji”.
3. **Po Etapie A:** commit `ETAP_A_WYBOR.json` (decyzja o E_max).
4. **Po Etapie B:** commit i znacznik czasu `ETAP_B_WYBOR.json`, przed Etapem C.
5. **Po treningu Etapu C:** commit hashy checkpointów, przed ewaluacją testową (§5.6).
6. **`ODSTEPSTWA.md`:** każda zmiana po zamrożeniu (data, treść, powód, czy przed czy po ewaluacji testowej).

---

## 11. Budżet

### 11.1. Limit finansowy (D1)

Twardy limit to **38 USD łącznie**: saldo Vast.ai 13.99 USD plus doładowanie do 100 PLN (≈ 24 USD po przewalutowaniu).

| Pozycja | Pod-limit planistyczny |
|---|---|
| Mini-pilot, pomiary, testy §9 | 5 USD |
| Etapy A + B | 12 USD |
| Etap C + E2 + E9 | 18 USD |
| Rezerwa (awarie, powtórki) | 3 USD |

Pod-limity służą do planowania. Wiążący jest wyłącznie limit łączny.

Przekroczenie pod-limitu wymaga ponownego oszacowania kosztu pozostałych etapów na podstawie zmierzonego zużycia (por. R4-11). Jeśli prognoza łączna przekracza 38 USD, badanie zostaje wstrzymane (§11.3).

### 11.2. Obciążenie (do przeliczenia po mini-pilocie)

| Element | Liczba biegów |
|---|---|
| Mini-pilot | 8 ramion × 10 ziaren × 2 zbiory = 160 |
| Etap A | 18 × 16 × 2 = 576 FF + 2 × 16 × 2 = 64 BP |
| Etap B | 18 × 9 × 2 = 324 FF + 2 × 9 × 2 = 36 BP; reguła brzegowa ≤ 18 × 12 × 2 = 432 FF + ≤ 2 × 12 × 2 = 48 BP |
| Etap C | 18 × N × 2 (N = 30: 1 080; N = 60: 2 160) |
| E2 | 2 × 10 × 2 = 40 (strojenie zawarte w Etapie A) |
| E9 | 6 × 3 × 10 × 2 = 360 (małe sieci) |
| E10 (lokalnie, CPU) | 7 ramion × 4 budżety × (8 strojenie + 10 ziaren) = 504 biegi poniżej 5 000 parametrów |

Biegi E2 i E9 w Etapie C liczą się do pod-limitu „Etap C + E2 + E9”.

**Po mini-pilocie** cała tabela jest przeliczana na koszt w USD dla każdego N oraz dla scenariusza najgorszego (N = 60 i dwukrotne podniesienie E_max według §5.2). Wynik trafia do protokołu przed zamrożeniem.

### 11.3. Zasady

**Przed zamrożeniem:** jeśli prognoza kosztu wszystkich etapów i analiz GPU przy N z §7.5 przekracza 33 USD, stosowane są cięcia w tej kolejności:
1. pominięcie E9;
2. pominięcie E2;
3. Etap A z 8 punktami Sobola (pierwsze 8).

N nie jest redukowane. Jeśli koszt nadal przekracza limit, badanie nie startuje, a autor decyduje o budżecie.

**Po zamrożeniu:** przekroczenie prognozy oznacza **wstrzymanie** badania, nie skrócenie. Decyzja autora o dodatkowym budżecie trafia do `ODSTEPSTWA.md`. Jeśli badanie nie zostanie wznowione, obowiązuje §5.6 pkt 6.

**Liczby niezmienne po zamrożeniu:** nie wolno zmniejszać N, liczby ramion ani E_max.

---

## 12. Identyfikowalność

Numery sekcji w 12.1–12.4 odnoszą się do wersji wskazanej w nagłówku tabeli.

### 12.1. Audyt → v2

| ID | Naprawa |
|---|---|
| S1, S5 | §5.2, §10 |
| S2, K16 | §7.3, §7.5, §9.2 |
| S3, S8, K11 | §7.2 |
| S4, G3 | §9.4 |
| S6 | §3 |
| S7, L5, K1, K2 | §5.3–5.4 |
| S9, K14, K17 | §6.2 |
| S10, K12, G5 | §9.5 |
| L1 | §3.1, §9.1.2 |
| L2, K10 | §3.3 |
| L3 | audyt |
| L4, K15 | §6.2 |
| L6, G2 | §1 |
| L7, K18 | §4.3 |
| K3 | §5.6, §7.5, §8 |
| K4, K5 | §2, §7.4, §9.4 |
| K6 | §7.3 |
| K7 | §7.7 |
| K8 | §3.3, E2 |
| K9 | §4.2, §5.3–5.6 |
| K13 | §4.2–4.3, §7.8 |
| G1 | §10.2 |
| G4 | §9.1–9.2 |
| G6 | §14 |

### 12.2. R1 → v2

Pełna tabela w `odpowiedzi/O1.md`. Wszystkie zarzuty zamknięte w R2 i R3.

### 12.3. R2 → v3, R3 → v4

Pełne tabele w `odpowiedzi/O2.md` i `odpowiedzi/O3.md`. Wszystkie zarzuty zamknięte w R3 i R4.

### 12.4. R4 i decyzje D2, D3 → v5

| ID | Naprawa w v5 |
|---|---|
| R4-01 | §9.3 (ramiona wewnątrzgrupowe), §7.5 (σ_d = 1.29 × max SD wewnątrzgrupowych, ρ̂) |
| R4-02 | §4.1, §7.8(d), §13.10 |
| R4-03 | §7.4 (zdania opisowe, reguła listy komparatorów, mechanizm tylko w części eksploracyjnej), §1 |
| R4-04 | §8 (synchronizacja po każdym biegu) |
| R4-05 | §9.1.9 (tolerancja), §9.1.6 i §8 (test na każdej instancji), §4.3 (generatory CPU, SHA-256) |
| R4-06 | §5.6 pkt 4 (definicja błędu kodu), §5.5 (`zaslepione/`) |
| R4-07 | §5.2 (reguła po A i po B, każda warstwa) |
| R4-08 | §7.6 |
| R4-09 | §7.4 (NIESPÓJNY-ZGODNY-KIERUNEK, wariant (a)), §13.11 |
| R4-10 | §4.1, §9.1.10, §10.1 |
| R4-11 | §7.2 (kontrola per typ), §11 (E2, pod-limity, cięcia), §5.6 pkt 6 (przerwanie), §7.4 (kierunki R+), usunięcie nieaktualnych godzin CPU |
| D2 | §1, §2 (H1, m = 6), §3.3, §7.4, §7.5 (m = 6), E7 |
| D3 | §1, §3.4, §4.4, §7.7 (E9, E10), §7.9, §11.2 |

### 12.5. R5 (weryfikacja) → v5 (poprawki bez kolejnej rundy)

| ID | Naprawa |
|---|---|
| R5-01 | §1, §7.4, §7.7: „bloki referencyjne” |
| R5-02 | §5.4, §11.2: BP w Etapie B |
| R5-03 | §4.3, §5.7, §7.7: E10 – dolna mediana, ziarno strojenia 50000, strojenie per (ramię, budżet) |
| R5-04 | §7.7 (E9): kryterium najlepszego ramienia referencyjnego |
| R5-05 | §7.4: zdanie NIESPÓJNY-ZGODNY-KIERUNEK |
| R5-06 | `DECYZJE_AUTORA.md` D1 zsynchronizowane z §11.1 |

### 12.6. Zmiany przed zamrożeniem, wynikające z implementacji (23.10.2026)

| Zmiana | Powód |
|---|---|
| §4.1, §5.7: dane kopiowane na instancję z weryfikacją hashy zamiast wbudowania w obraz Dockera; oficjalny obraz PyTorch z ustaloną wersją | wbudowanie danych wymagałoby własnego rejestru obrazów; weryfikacja hashy przy każdym wczytaniu daje tę samą gwarancję |
| §5.7 (E10): E_max 2 000 → 5 000, cierpliwość 100 → 1 000 | test wykonalności na ziarnie strojenia 50000 (spoza ziaren ewaluacyjnych): FF na spirali uczy się z długimi plateau; przy cierpliwości 100 wszystkie ramiona FF zatrzymywały się na ~65% (BP: 100%), przy cierpliwości 1 000 część konfiguracji osiąga 99–100%. Bez tej zmiany rysunek E10 byłby artefaktem zbyt wczesnego zatrzymania |
| §13.14: ograniczenie F-MNIST | wynik testu 9.1.8 (38.9% > 1%), zgodnie z regułą §9.1.8 |
| §15: mapowanie protokołu na kod | identyfikowalność implementacji |

---

## 13. Ograniczenia znane przed badaniem

1. Tylko MLP i małe zbiory obrazów w skali szarości.
2. Jeden schemat FF: etykieta w pikselach, losowe negatywy, uczenie zachłanne, suma dobroci z obu warstw.
3. Ramiona K uczą się lokalnie w blokach, nie czysto FF warstwa po warstwie (Nøkland i Eidnes, 2019; Belilovsky i in., 2019).
4. **Strojenie:**
   - wyniki zależą od przestrzeni i budżetu strojenia (16 punktów, wspólne c dla obu warstw FF);
   - dla wyjść ReLU i LeakyReLU c jest częściowo wymienne ze skalą wag;
   - dla wyjść tanh wartości c bliskie 1 nasycają neurony;
   - krajobraz strojenia jest raportowany (E5).
5. **Ograniczona dobroć tanh.** Ramiona z wyjściem tanh mają G ≤ w. W H1 trzy komparatory kończą się tanh (`K-T`, `K-LLT`, `K-GGT`), a trzy nie (`K-R`, `K-L`, `K-G`). Przegrana LGT z komparatorem o nieograniczonym wyjściu może odzwierciedlać typ wyjścia (E7).
6. Komparatory są oceniane wszystkie, bez selekcji.
7. Stały zbiór testowy; wrażliwość sprawdzana w §7.8(a) i (d).
8. Inicjalizacja domyślna PyTorch, nieoptymalna dla części nieliniowości.
9. W BANK jednostki tanh mają ograniczony wkład do Σa² (E8).
10. Zbiór testowy MNIST był oglądany w pilotach poprzedzających wybór hipotez. Obciążenie jest znikome; QMNIST-50k służy jako kontrola (§7.8(d)).
11. Przy szerokim S na Fashion-MNIST mały, zgodny kierunkowo efekt może dać NIESPÓJNY-ZGODNY-KIERUNEK zamiast klasy P (§7.4).
12. **E9:** konfiguracje przeniesione z w = 250 bez ponownego strojenia; θ skaluje się jak c · w. Mniejsze sieci mogą być przez to dostrojone gorzej, nierówno między ramionami.
13. **E10:** zadanie syntetyczne 2D, wejście FF z etykietą konkatenowaną, dopasowanie parametrów łączne (nie per warstwa). Wynik jest ilustracją, nie dowodem.
14. **Nakładanie etykiety na Fashion-MNIST** (test 9.1.8): 38.9% obrazów treningowych F-MNIST ma niezerowe piksele w pozycjach 0–9 (MNIST: 0%). Nadpisanie tych pikseli etykietą usuwa część informacji z górnej krawędzi obrazu. Dotyczy jednakowo wszystkich ramion FF; ramiona BP widzą obraz bez zmian, co zawyża lukę FF–BP w E2 na F-MNIST.

---

## 14. Literatura

- Belilovsky, E., Eickenberg, M., & Oyallon, E. (2019). Greedy layerwise learning can scale to ImageNet. *ICML 2019*, PMLR 97, 583–593.
- Berger, R. L., & Hsu, J. C. (1996). Bioequivalence trials, intersection-union tests and equivalence confidence sets. *Statistical Science*, 11(4), 283–319.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281–305.
- Bouthillier, X., i in. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems 3 (MLSys 2021)*.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS 2018*.
- Dodge, J., Gururangan, S., Card, D., Schwartz, R., & Smith, N. A. (2019). Show your work: Improved reporting of experimental results. *EMNLP-IJCNLP 2019*, 2185–2194.
- Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv:1606.08415*.
- Hinton, G. (2022). The Forward-Forward algorithm: Some preliminary investigations. *arXiv:2212.13345*.
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Melis, G., Dyer, C., & Blunsom, P. (2018). On the state of the art of evaluation in neural language models. *ICLR 2018*.
- Montúfar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the number of linear regions of deep neural networks. *NeurIPS 2014*.
- Nøkland, A., & Eidnes, L. H. (2019). Training neural networks with local error signals. *ICML 2019*, PMLR 97, 4839–4850.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. *arXiv:1708.07747*.
- Yadav, C., & Bottou, L. (2019). Cold case: The lost MNIST digits. *NeurIPS 2019*.
- Yarotsky, D. (2017). Error bounds for approximations with deep ReLU networks. *Neural Networks*, 94, 103–114.

---

## 15. Implementacja (katalog `kod/`)

| Plik | Sekcje protokołu |
|---|---|
| `ffexp/streams.py` | §4.3 strumienie losowe |
| `ffexp/data.py` | §4.1–4.4 dane, hashe, podziały, blokada zbioru testowego, spirala |
| `ffexp/models.py` | §3 ramiona, nieliniowości, topologie, liczby parametrów |
| `ffexp/train.py` | §5.1–5.2, §5.7, §7.6 uczenie FF i BP, wczesne zatrzymanie, NaN |
| `ffexp/plans.py` | §5.2–5.5 plany etapów, reguła E_max, wybór top-3, reguła brzegowa, wybór konfiguracji |
| `ffexp/runner.py` | §5.5, §8 wykonanie, zapis atomowy, zaślepienie (`zaslepione/`) |
| `ffexp/monitor.py` | §8 monitor (wyłącznie liczby biegów) |
| `ffexp/sync_verify.py`, `vast/*.sh` | §8 wdrożenie i synchronizacja z weryfikacją hashy |
| `ffexp/gpu_check.py` | §9.1.6, §9.1.9 na docelowym GPU; pomiar przepustowości (§9.3) |
| `ffexp/minipilot.py` | §9.3, §7.5, §5.2, §11.2 wyprowadzenie N, S, σ_d, ρ, E_max, kosztu |
| `ffexp/analysis.py` | §7.2–7.5 reguły decyzyjne, łączenie zbiorów, symulacja mocy |
| `ffexp/selftest_analysis.py` | §9.2 test potoku analizy (z niezależną implementacją referencyjną) |
| `ffexp/evaluate_test.py` | §5.6 ewaluacja testowa bramkowana hashami |
| `ffexp/final.py` | §7.3–7.8 analiza konfirmacyjna, wrażliwość, analizy eksploracyjne |
| `ffexp/spiral.py` | E10, §7.9 rysunki |
| `tests/test_protokol.py` | §9.1.1–9.1.10 |

