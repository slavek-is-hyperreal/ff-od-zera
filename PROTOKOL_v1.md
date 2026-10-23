# Protokół badania: heterogeniczność i kolejność nieliniowości w sieciach Forward-Forward przy stałym budżecie parametrów

**Wersja:** v1 (projekt, NIE zamrożony)
**Data:** 23.10.2026
**Autor:** Sławomir Majcher
**Katalog:** `/vectorlegis_ssd_pool/uczelnia/eksperyment_ff_od_zera/`
**Dokumenty powiązane:** `00_AUDYT_BLEDOW.md` (lista błędów poprzednich eksperymentów; identyfikatory S*, L*, K*, G*)

Ten protokół zastępuje w całości protokoły z `eksperyment_sca_ff`, `eksperyment_laminar_ff` i `eksperyment_kontrola_wymiaru_ff`. Wyniki tamtych eksperymentów służą wyłącznie jako dane pilotowe (analiza mocy, szacunek czasu). Nie są cytowane jako wyniki.

---

## 1. Pytanie badawcze i zakres

**Pytanie główne.** W sieciach MLP uczonych algorytmem Forward-Forward (FF), przy równym budżecie parametrów, równej głębokości i równym budżecie strojenia hiperparametrów:

1. Czy blok złożony z różnych nieliniowości ułożonych kolejno (LeakyReLU → GELU → tanh, dalej „kaskada LGT”) daje wyższą dokładność niż najlepszy blok jednorodny (jedna nieliniowość we wszystkich podwarstwach)?
2. Czy liczy się konkretna kolejność LGT, czy wystarcza sama różnorodność nieliniowości w głębokości?
3. Czy w płaskiej sieci FF (jedna warstwa liniowa na warstwę FF) równoległy bank różnych nieliniowości daje wyższą dokładność niż najlepsza pojedyncza nieliniowość?

**Czego badanie nie rozstrzyga:** zachowania na danych innych niż MNIST i Fashion-MNIST, architektur splotowych, innych schematów generowania negatywów, trafności analogii z budową kory mózgowej. Analogia biologiczna była motywacją, nie jest przedmiotem testu (por. L6).

**Status badania:** konfirmacyjne dla H1–H3 (§2), eksploracyjne dla wszystkich pozostałych analiz (§7.6).

---

## 2. Hipotezy konfirmacyjne

Wszystkie kontrasty są parowane po ziarnie (§4.3). Metryka główna: dokładność na zbiorze testowym (%). Δ = dokładność ramienia badanego − dokładność komparatora (p.p.).

| ID | Ramię badane | Komparator | Przewidywanie autora |
|---|---|---|---|
| **H1** | `K-LGT` | `K-MONO*`: najlepsze z {`K-R`, `K-L`, `K-G`, `K-T`}, wybrane na zbiorze walidacyjnym w Etapie B (§5.4) | Δ > 0 |
| **H2** | `K-LGT` | `K-ALT*`: najlepsze z {5 pozostałych permutacji L, G, T; `K-BANK`}, wybrane w Etapie B | Δ > 0 |
| **H3** | `P-BANK` | `P-MONO*`: najlepsze z {`P-R`, `P-L`, `P-G`, `P-T`}, wybrane w Etapie B | Δ > 0 |

Komparatorem jest zawsze **najsilniejszy** konkurent wybrany na danych niezależnych od danych konfirmacyjnych (inne ziarna, bez zbioru testowego). Usuwa to zarzut porównywania ze słabym wariantem (por. K10, K8).

Przewidywanie autora jest kierunkowe, ale testy są dwustronne, a wynik klasyfikowany jest do jednej z czterech kategorii (§7.3). Wynik przeciwny przewidywaniu jest pełnoprawnym wynikiem i zostanie zaraportowany tak samo jak zgodny (por. K4, K5).

**Interpretacja łączna (ustalona z góry):**
- H1 przewaga ∧ H2 przewaga → efekt specyficzny dla kolejności LGT.
- H1 przewaga ∧ H2 równoważność → efekt różnorodności nieliniowości w głębokości, kolejność nieistotna.
- H1 równoważność lub gorszość → brak korzyści z kaskady względem najlepszej monokultury.
- H3 niezależnie: korzyść (lub jej brak) z różnorodności w płaskiej warstwie.

---

## 3. Ramiona eksperymentu

### 3.1. Topologie

**Płaska (P), k = 1:** dwie warstwy FF, każda = jedna warstwa liniowa + nieliniowość.
- Warstwa FF 1: 784 → 250 (196 250 parametrów)
- Warstwa FF 2: 250 → 250 (62 750 parametrów)
- Razem: **259 000**

**Blokowa (K), k = 3:** dwie warstwy FF, każda = blok trzech podwarstw (liniowa + nieliniowość w każdej podwarstwie, brak złożeń liniowo-liniowych, por. L1). Wewnątrz bloku gradient straty FF propaguje się przez trzy podwarstwy. Między warstwami FF gradient nie przechodzi (`detach`).
- Blok 1: 784 → 163 → 163 → 250 (195 687 parametrów; −0.29% względem P)
- Blok 2: 250 → 103 → 103 → 250 (62 565 parametrów; −0.29% względem P)
- Razem: **258 252** (−0.29%)

Wymiary podwarstw dobrano tak, aby **każda warstwa FF osobno** miała ten sam budżet co odpowiadająca jej warstwa płaska. Usuwa to wąskie gardło 250 → 60 z pilota.

### 3.2. Nieliniowości

- R = ReLU
- L = LeakyReLU(ujemne nachylenie 0.5)
- G = GELU (dokładne, `approximate='none'`)
- T = tanh
- BANK = jednostki warstwy (podwarstwy) podzielone na trzy ciągłe grupy: L | G | T, liczności ⌈n/3⌉, ⌈(n−1)/3⌉, ⌊n/3⌋ (dla 250: 84/83/83; dla 163: 55/54/54; dla 103: 35/34/34)

Wyjście ostatniej podwarstwy bloku jest wyjściem warstwy FF (z niego liczona jest dobroć).

### 3.3. Lista ramion (18)

| Ramię | Topologia | Nieliniowości (podwarstwa 1 → 2 → 3) | Rola | Izolowany czynnik |
|---|---|---|---|---|
| `P-R` | P | R | komparator H3 | odniesienie do FF Hintona |
| `P-L` | P | L | komparator H3 | monokultura składnika |
| `P-G` | P | G | komparator H3 | monokultura składnika |
| `P-T` | P | T | komparator H3 | monokultura składnika |
| `P-BANK` | P | BANK | badane H3 | różnorodność w szerokości |
| `K-R` | K | R→R→R | komparator H1 | monokultura w głębokości |
| `K-L` | K | L→L→L | komparator H1 | j.w. |
| `K-G` | K | G→G→G | komparator H1 | j.w. |
| `K-T` | K | T→T→T | komparator H1 | j.w. |
| `K-LGT` | K | L→G→T | badane H1, H2 | hipoteza kaskady |
| `K-LTG` | K | L→T→G | komparator H2 | kolejność |
| `K-GLT` | K | G→L→T | komparator H2 | kolejność |
| `K-GTL` | K | G→T→L | komparator H2 | kolejność |
| `K-TLG` | K | T→L→G | komparator H2 | kolejność |
| `K-TGL` | K | T→G→L | komparator H2 | kolejność (odwrócona) |
| `K-BANK` | K | BANK→BANK→BANK | komparator H2 | różnorodność w głębokości bez kolejności |
| `BP-P-R` | P + odczyt 250→10 | R | eksploracyjne | luka do backpropagacji, topologia P |
| `BP-K-LGT` | K + odczyt 250→10 | L→G→T | eksploracyjne | luka do backpropagacji, topologia K |

Ramiona BP uczone są end-to-end entropią krzyżową na obrazach bez nałożonej etykiety, z tą samą topologią co odpowiadające ramię FF (por. K8). Parametry: 261 510 (`BP-P-R`), 260 762 (`BP-K-LGT`).

**Ramię usunięte:** FF ReLU D = 1000 (1.786 M parametrów) z pilota. Nie dotyczy żadnej hipotezy, kosztuje ok. 4× więcej czasu niż ramię P, a pilot pokazał różnicę 0.16 p.p. względem D = 250.

---

## 4. Dane

### 4.1. Zbiory
- **MNIST** (LeCun i in., 1998): 60 000 treningowych, 10 000 testowych.
- **Fashion-MNIST** (Xiao i in., 2017): 60 000 / 10 000.

Oba zbiory są konfirmacyjne. Fashion-MNIST zmniejsza efekt sufitu (MNIST w pilocie: 95–98%). Piksele skalowane do [0, 1], bez augmentacji, obraz spłaszczony do 784.

### 4.2. Podziały
- Zbiór testowy (10 000) jest używany **wyłącznie** w Etapie C, raz na bieg. Runner w trybie Etapu A/B nie ma ścieżki kodu do zbioru testowego (wymuszone testem jednostkowym, §9).
- Zbiór treningowy (60 000) dzielony losowo na 50 000 treningowych / 10 000 walidacyjnych, osobno dla każdego ziarna (§4.3). W ten sposób wnioskowanie obejmuje też zmienność próby danych, nie tylko inicjalizacji (por. K13; Bouthillier i in., 2021).

### 4.3. Ziarna i parowanie
Ziarno s determinuje cztery **niezależne** strumienie losowe (`torch.Generator` seedowany z (s, id_strumienia)):
1. `split` – podział 50k/10k
2. `order` – kolejność batchy
3. `neg` – etykiety negatywne
4. `init` – inicjalizacja wag

Strumienie 1–3 są identyczne dla wszystkich ramion przy danym s. Różnice w liczbie parametrów nie przesuwają strumieni danych. To jest podstawą parowania.

Pule ziaren (rozłączne z pilotem, por. K18, L7):
- Etap A: s = 30001 (jedno ziarno, podział stały)
- Etap B: s ∈ {20001, 20002, 20003}
- Etap C: s ∈ {10001, …, 10040} (N = 40, §7.5)

---

## 5. Procedura uczenia

### 5.1. Forward-Forward (ramiona P i K)
- **Kodowanie etykiety** (Hinton, 2022): pierwsze 10 pikseli zerowane, piksel o indeksie y ustawiany na 1.0.
- **Negatywy:** ten sam obraz z etykietą y' losowaną jednostajnie z {0..9} \ {y} (strumień `neg`), losowaną na nowo w każdym batchu.
- **Dobroć:** G(a) = Σⱼ aⱼ², suma po jednostkach wyjścia warstwy FF.
- **Strata warstwy FF:** L = mean[softplus(θ − G_pos)] + mean[softplus(G_neg − θ)].
- **Próg:** θ = c · 250, gdzie c to hiperparametr strojony (§5.3), wspólny dla obu warstw FF danego ramienia.
- **Wejście warstwy 2:** wyjście warstwy 1 znormalizowane L2: a / (‖a‖₂ + 1e-8).
- **Kolejność uczenia:** zachłanna. Warstwa 1 uczona do zatrzymania (§5.2), zamrażana, potem warstwa 2.
- **Optymalizator:** AdamW, batch 128 (`drop_last=True`), stały współczynnik uczenia.
- **Klasyfikacja:** ŷ = argmax_c Σₗ Gₗ(x z etykietą c). Ponieważ θ jest wspólne dla warstw, odjęcie θ nie zmienia argmax.

### 5.2. Liczba epok i zatrzymanie
- Maksimum: **30 epok na warstwę FF**.
- Wczesne zatrzymanie z cierpliwością 5 epok na metryce walidacyjnej:
  - warstwa 1: dokładność klasyfikacji z samej G₁
  - warstwa 2: dokładność z G₁ + G₂
- Przywracane są wagi z najlepszej epoki.
- Ten sam maksymalny budżet dla wszystkich ramion FF (por. S1, S5).
- **BP:** maksimum 60 epok, cierpliwość 5, metryka: dokładność walidacyjna.

### 5.3. Etap A – strojenie hiperparametrów z równym budżetem
- **Przestrzeń:**
  - lr ∈ [1e-4, 1e-2] (log)
  - wd ∈ [1e-6, 1e-2] (log)
  - c ∈ [0.02, 2] (log), czyli θ ∈ [5, 500]
- **Punkty:** 20 punktów z sekwencji Sobola (scrambled, ziarno 0), **identycznych dla każdego ramienia FF**. Ramiona BP używają tych samych 20 par (lr, wd).
- Każdy punkt: jeden bieg na ziarnie 30001, kryterium = dokładność walidacyjna.
- Osobno dla każdego zbioru danych.

Uzasadnienie: identyczne punkty i równy budżet eliminują przewagę wynikającą z lepszego dostrojenia θ (por. K1, K2, S7, L5; Melis i in., 2018; Dodge i in., 2019). Strojenie losowe/quasi-losowe zgodnie z Bergstra i Bengio (2012).

### 5.4. Etap B – wybór konfiguracji i komparatorów
- Dla każdego ramienia i zbioru: 3 najlepsze punkty z Etapu A × 3 ziarna {20001, 20002, 20003}.
- **Konfiguracja ramienia** = punkt o najwyższej średniej dokładności walidacyjnej (remis: niższe c).
- **Komparatory** `K-MONO*`, `K-ALT*`, `P-MONO*` = ramię o najwyższej średniej dokładności walidacyjnej (przy wybranej konfiguracji) w danej grupie, **osobno dla każdego zbioru danych**.
- Wynik Etapu B (konfiguracje + komparatory) zapisywany do `ETAP_B_WYBOR.json`, commitowany i hashowany **przed** startem Etapu C (§10).

### 5.5. Etap C – biegi konfirmacyjne
- 18 ramion × 40 ziaren × 2 zbiory = **1440 biegów** z konfiguracjami z Etapu B.
- Każdy bieg: trening na 50k, wczesne zatrzymanie na 10k walidacji, **jedna** ewaluacja na zbiorze testowym.

### 5.6. Backpropagacja (ramiona BP)
Entropia krzyżowa, AdamW, batch 128. Wejście: obraz bez nałożonej etykiety. Odczyt liniowy z wyjścia ostatniej warstwy.

---

## 6. Metryki

### 6.1. Główna
Dokładność na zbiorze testowym (%) z §5.1 (FF) lub argmax logitów (BP).

### 6.2. Drugorzędne (wyłącznie opisowe, §7.6)

Liczone na pozytywach zbioru walidacyjnego, dla wyjścia każdej warstwy FF.

- **Udział jednostek nieaktywnych:** jednostka j jest nieaktywna, gdy sd(aⱼ) < 0.01 · medianaₖ sd(aₖ) w tej samej warstwie. Definicja względna, niezależna od skali aktywacji (por. S9, K14). Bez porównań z wartościami z pilota.
- **Wymiar efektywny (participation ratio):** PR = (Σλᵢ)² / Σλᵢ², λ = wartości własne macierzy kowariancji wyjść warstwy.
- **Krzywe uczenia:** strata i dokładność walidacyjna co epokę (por. K17).
- **Liczba epok do zatrzymania** oraz **czas ściany** – tylko opis, bez hipotez (por. L4, K15).

---

## 7. Analiza statystyczna

### 7.1. Jednostka analizy
Ziarno s (N = 40 na zbiór). Dla każdego kontrastu i zbioru: dᵢ = Acc_badane(sᵢ) − Acc_komparator(sᵢ).

### 7.2. Test
- Parowany test t dla średniej różnic (dwustronny).
- Przedział ufności: t-Studenta.
- **Bez wstępnego testu normalności i bez warunkowego wyboru testu** (por. K11).
- Analiza wrażliwości (raportowana, niewiążąca): test Wilcoxona i przedział bootstrap BCa (10 000 próbek).

### 7.3. Klasyfikacja wyniku (na zbiór danych)
Poziom α = 0.05 z korektą Bonferroniego na 3 hipotezy: **α' = 0.0167**.

| Wynik | Warunek |
|---|---|
| **Przewaga** | dolna granica (1 − α') PU dla Δ > 0 |
| **Gorszość** | górna granica (1 − α') PU dla Δ < 0 |
| **Równoważność** | (1 − 2α') PU dla Δ zawiera się w (−SESOI, +SESOI) (TOST; Schuirmann, 1987; Lakens, 2017) |
| **Nierozstrzygnięty** | żadne z powyższych |

Przewaga i równoważność mogą wystąpić jednocześnie. Wtedy wynik brzmi „przewaga mniejsza niż SESOI”. Dodatkowo raportowane jest, czy dolna granica PU przekracza SESOI („przewaga istotna praktycznie”).

### 7.4. Łączenie zbiorów danych
Hipoteza otrzymuje werdykt X ∈ {przewaga, gorszość, równoważność} tylko wtedy, gdy X zachodzi na **obu** zbiorach. Jest to test przecięcia-sumy (Berger i Hsu, 1996), który nie wymaga dodatkowej korekty α. W przeciwnym razie werdykt brzmi „niespójny między zbiorami” i podawane są oba wyniki.

### 7.5. SESOI i liczebność próby
- **SESOI:** MNIST 0.30 p.p., Fashion-MNIST 0.75 p.p. Obie wartości odpowiadają ok. 10% względnej redukcji błędu przy błędzie bazowym z pilota (ok. 3%) i z literatury dla MLP na Fashion-MNIST (ok. 8–12%).
- **Moc:** SD różnic parowanych w pilocie wyniosło 0.15–0.40 p.p. (`00_AUDYT_BLEDOW.md`). Losowanie podziału danych może je zwiększyć, więc przyjęto σ_d = 0.50 dla MNIST.
  - Przy N = 40, α' = 0.0167, Δ = SESOI: moc testu przewagi 0.90, moc TOST przy Δ = 0 wynosi 0.88 (symulacja, 20 000 powtórzeń).
  - Dla σ_d = 0.40: odpowiednio 0.985 / 0.987.
  - Dla Fashion-MNIST przyjęto σ_d = 1.25 (proporcjonalnie do SESOI), co daje tę samą moc.
- **Brak analiz pośrednich.** N = 40 jest stałe. Runner nie liczy statystyk porównawczych przed zakończeniem wszystkich 1440 biegów (por. K3). Monitor postępu pokazuje wyłącznie liczbę ukończonych biegów i flagi błędów.

### 7.6. Analizy eksploracyjne (bez wniosków konfirmacyjnych, bez korekty, opisane jako eksploracyjne)
- **E1 – efekt głębokości:** `K-x` vs `P-x` dla x ∈ {R, L, G, T, BANK}.
- **E2 – luka do BP:** `BP-P-R` − `P-R` oraz `BP-K-LGT` − `K-LGT`.
- **E3:** `K-LGT` vs każda monokultura i każda permutacja osobno.
- **E4:** metryki drugorzędne (§6.2) dla wszystkich ramion.
- **E5:** krajobraz strojenia z Etapu A: dokładność walidacyjna w funkcji c dla każdego ramienia (czy θ tłumaczy wyniki pilota).
- **E6:** klasyfikacja tylko z ostatniej warstwy FF.

### 7.7. Wykluczenia i błędy
- Żaden bieg nie jest wykluczany.
- Bieg, w którym strata przyjmie wartość NaN, zatrzymuje się. Wynikiem jest model z najlepszej dotychczasowej epoki, a gdy takiej nie ma, dokładność 10% (losowa). Liczba takich biegów jest raportowana per ramię.
- Awaria wynikająca z błędu w kodzie zatrzymuje całe badanie. Po poprawce **wszystkie** biegi dotkniętych ramion są powtarzane, a zdarzenie trafia do `ODSTEPSTWA.md`.

---

## 8. Wykonanie

- **Jeden typ sprzętu** dla wszystkich biegów danego etapu. Zapis: CPU/GPU, wersje Pythona, PyTorch, CUDA, sterowników, hash commitu.
- `torch.use_deterministic_algorithms(True)`, stała liczba wątków. Test powtarzalności w §9.
- **Kolejność biegów:** dla każdego ziarna ramiona i zbiory w kolejności losowej (permutacja z ziarna), aby zmiany warunków maszyny w czasie nie korelowały z ramieniem.
- **Biegi atomowe:** wynik zapisywany po ukończeniu biegu. Przerwany bieg uruchamiany jest od zera z tym samym ziarnem (dopuszczalne np. przy przerwie na wykład).
- Wszystkie surowe wyniki (JSON per bieg, krzywe uczenia) są archiwizowane.

---

## 9. Weryfikacja przed zamrożeniem (warunek zamrożenia)

Kod (runner + analiza) jest pisany **po** zaakceptowaniu tego protokołu i na jego podstawie (por. G4). Przed zamrożeniem muszą przejść:

### 9.1. Testy jednostkowe
1. Liczba parametrów każdego ramienia = tabela §3.
2. Każda warstwa liniowa (oprócz odczytu BP) ma po sobie nieliniowość (por. L1).
3. Etykieta negatywna ≠ prawdziwa (100% przypadków).
4. Przy uczeniu warstwy 2 gradienty parametrów warstwy 1 są zerowe/nieobecne.
5. W trybie Etapu A/B nie da się wczytać zbioru testowego.
6. Ten sam (ramię, ziarno) daje bitowo identyczny wynik dwukrotnie.
7. Strumienie `split`/`order`/`neg` dają identyczne dane dla dwóch różnych ramion przy tym samym ziarnie.

### 9.2. Test skryptu analizy na danych symulowanych
- (a) H₀ prawdziwa (Δ = 0): odsetek fałszywych „przewag” ≤ α' (2000 symulacji).
- (b) Δ = SESOI: odsetek „przewag” zgodny z mocą z §7.5.
- (c) Δ = 0: odsetek „równoważności” zgodny z mocą TOST.

### 9.3. Pomiar czasu
- Biegi na danych losowych (szum, nie MNIST), 1 epoka na ramię, na docelowym sprzęcie.
- Ekstrapolacja kosztu §11. Jeśli budżet zostanie przekroczony, stosuje się redukcję z §11.2, decyzja jest zapisana przed zamrożeniem.

### 9.4. Szablon raportu (results-blind)
`RAPORT_SZABLON.md` z miejscami na wszystkie liczby, wszystkie hipotezy i wszystkie cztery możliwe werdykty każdej z nich. Raport końcowy wypełnia szablon. Interpretacje spoza szablonu są oznaczane jako eksploracyjne (por. G3, S4).

### 9.5. Zasady języka raportu
- Bez słów „dowodzi”, „bezprecedensowy”, „idealny”, „rozwiązuje”.
- Każdy wniosek z zastrzeżeniem zakresu: „MLP, MNIST/Fashion-MNIST, schemat FF z §5”.
- Wynik negatywny opisywany wprost (por. K12, G5).

---

## 10. Zamrożenie i znacznik czasu

1. Po akceptacji protokołu i przejściu §9:
   - commit z tagiem `ff-od-zera-prereg`,
   - plik `SHA256SUMS` (protokół, runner, skrypt analizy, szablon raportu).
2. **Zewnętrzny znacznik czasu** (por. G1): rejestracja na OSF lub publikacja tagu w publicznym repozytorium. Decyzja należy do autora, przed Etapem A.
3. Po Etapie B: commit `ETAP_B_WYBOR.json` + hash, przed startem Etapu C. Jeśli dostępny jest zewnętrzny znacznik czasu, ten sam wymóg dotyczy tego commitu.
4. Każda zmiana po zamrożeniu trafia do `ODSTEPSTWA.md`: data, treść, powód, czy nastąpiła przed czy po wglądzie w wyniki Etapu C.

---

## 11. Budżet obliczeniowy

### 11.1. Szacunek z pilota
Pilot: 3 rdzenie i5-3450, 15 epok na warstwę. Czasy: płaska ~200 s, blokowa ~245 s, BP ~95 s.

Przy maks. 30 epokach i wczesnym zatrzymaniu przyjmuję średnio ok. 20 epok (×1.33). Daje to ok. 87 minut na komplet 18 ramion dla jednego (ziarno, zbiór).

| Etap | Komplety (ziarno × zbiór) | CPU (3 rdzenie) |
|---|---|---|
| A: 20 punktów × 2 zbiory | 40 | ~58 h |
| B: 3 konfiguracje × 3 ziarna × 2 zbiory | 18 | ~26 h |
| C: 40 ziaren × 2 zbiory | 80 | ~116 h |
| **Razem** | | **~200 h** |

Na współdzielonym komputerze (zajęcia na Teams) jest to ok. 2–3 tygodnie biegów nocnych.

**Wariant zalecany:** wynajem GPU (np. Vast.ai). Szacunek: ~15–25 h GPU przy kilku biegach równolegle na jednej karcie, rząd kilku–kilkunastu USD. Wymaga potwierdzenia pomiarem (§9.3) i decyzji autora. Wszystkie biegi danego etapu na tym samym typie sprzętu (§8).

### 11.2. Redukcja (stosowana w tej kolejności, decyzja przed zamrożeniem)
1. Etap A: 20 → 12 punktów Sobola (pierwsze 12 z tej samej sekwencji).
2. Ramiona BP: z Etapu C do analizy pobocznej z N = 10.
3. N: 40 → 30 (moc przy σ_d = 0.40: 0.93; przy σ_d = 0.50: 0.77; raportowane jawnie).

---

## 12. Identyfikowalność: błąd z audytu → miejsce naprawy

| ID | Naprawa |
|---|---|
| S1, S5 | §5.2 (równy, wystarczający budżet epok + wczesne zatrzymanie), §10.4 (odstępstwa) |
| S2, K16 | §7.3, §7.5 (progi = SESOI uzasadnione, zapisane w kodzie analizy testowanym w §9.2) |
| S3 | §7.2 (test zawsze wykonywany) |
| S4, G3 | §9.4 (szablon results-blind) |
| S6 | §3 (wszystkie ramiona FF: 259 000 ± 0.3%) |
| S7, L5, K1, K2 | §5.3 (θ strojone z równym budżetem i identycznymi punktami) |
| S8 | §7.2 (SD z ddof=1, PU t-Studenta) |
| S9, K14 | §6.2 (definicja względna, tylko opisowa) |
| S10, K12, G5 | §9.5 |
| L1 | §3.1, §9.1 pkt 2 |
| L2 | §3.3 (głębokość jako osobny czynnik: P vs K; monokultury K) |
| L3 | `00_AUDYT_BLEDOW.md` (stan faktyczny opisany) |
| L4, K15 | §6.2 (czas tylko opisowo) |
| L6 | §1 (zakres) |
| L7, K18 | §4.3 (nowe pule ziaren) |
| K3 | §7.5 (brak analiz pośrednich, stałe N) |
| K4, K5 | §2 (wszystkie wyniki raportowane), §9.4 |
| K6 | §7.3 (jedna reguła decyzyjna, bez min z testów) |
| K7 | §7.6 (brak „dekompozycji”; kontrasty raportowane osobno) |
| K8 | §3.3 (BP o tej samej topologii; tylko eksploracyjnie) |
| K9 | §4.2, §5.3–5.5 (walidacja vs test) |
| K10 | §3.3 (permutacje, `K-BANK`, monokultury K) |
| K11 | §7.2 |
| K13 | §4.2–4.3 (losowy podział na ziarno) |
| K17 | §6.2 |
| G1 | §10.2 |
| G2 | §1 (pilot tylko do mocy i czasu) |
| G4 | §9.1–9.2 |
| G6 | §14 (krótka, zweryfikowana bibliografia) |

---

## 13. Ograniczenia (znane przed badaniem)

1. Tylko MLP i małe zbiory obrazów w skali szarości. Uogólnienie na CIFAR lub sieci splotowe nie jest testowane.
2. Jeden schemat FF: nakładanie etykiety, losowe negatywy, zachłanne uczenie warstw. Inne warianty FF (np. negatywy generowane, uczenie równoczesne) mogą dawać inne wyniki.
3. Ramiona K używają lokalnej backpropagacji wewnątrz bloku. Nie są „czystym” FF warstwa-po-warstwie, lecz uczeniem lokalnym blokowym (por. Nøkland i Eidnes, 2019; Belilovsky i in., 2019).
4. Wyniki są warunkowe względem przestrzeni i budżetu strojenia (20 punktów). Ramię o wąskim optimum może być niedoszacowane. Budżet jest jednak równy dla wszystkich ramion.
5. Komparatory „najlepszy z grupy” są wybierane na walidacji z 3 ziarnami. Błędny wybór działa na niekorzyść hipotezy (zachowawczo), nie na korzyść.
6. Zbiór testowy jest stały. Zmienność próby danych jest częściowo uchwycona przez losowanie podziału treningowego.

---

## 14. Literatura

- Belilovsky, E., Eickenberg, M., & Oyallon, E. (2019). Greedy layerwise learning can scale to ImageNet. *ICML 2019*, PMLR 97.
- Berger, R. L., & Hsu, J. C. (1996). Bioequivalence trials, intersection-union tests and equivalence confidence sets. *Statistical Science*, 11(4), 283–319.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281–305.
- Bouthillier, X., i in. (2021). Accounting for variance in machine learning benchmarks. *MLSys 2021*.
- Dodge, J., Gururangan, S., Card, D., Schwartz, R., & Smith, N. A. (2019). Show your work: Improved reporting of experimental results. *EMNLP-IJCNLP 2019*.
- Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv:1606.08415*.
- Hinton, G. (2022). The Forward-Forward algorithm: Some preliminary investigations. *arXiv:2212.13345*.
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Melis, G., Dyer, C., & Blunsom, P. (2018). On the state of the art of evaluation in neural language models. *ICLR 2018*.
- Nøkland, A., & Eidnes, L. H. (2019). Training neural networks with local error signals. *ICML 2019*, PMLR 97.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. *arXiv:1708.07747*.
