# R5 – ukierunkowana weryfikacja PROTOKOL_v5.md

**Recenzent:** R4 (niezależny), runda weryfikacyjna
**Data:** 23.10.2026
**Przedmiot:** `PROTOKOL_v5.md`, `odpowiedzi/O4.md`, `DECYZJE_AUTORA.md` (D1–D3)

**Obliczenia:** `/tmp/claude-1000/r5_power.py`. Skrypt używa tej samej implementacji §7.3–7.4 co w R4. Każda konfiguracja to 20 000 powtórzeń Monte Carlo, kontrola błędu I rodzaju – 40 000.

---

## 1. Werdykt

**DROBNE POPRAWKI. Protokół jest gotowy do zamrożenia po wprowadzeniu poprawek drobnych, bez kolejnej rundy.**

Bilans:
- **Zarzuty R4:** 10 usuniętych, 1 częściowo (R4-11).
- **Nowe zarzuty:** 0 KRYTYCZNYCH, 0 POWAŻNYCH, 6 DROBNYCH (R5-01 do R5-06).

---

## 2. Status zarzutów R4

| ID | Status | Uzasadnienie |
|---|---|---|
| R4-01 | **USUNIĘTY** | Mini-pilot ma 8 ramion i 6 kontrastów wewnątrzgrupowych. σ_d = 1.29 × max SD. Kierunek ρ = min(ρ̂, 0.5) jest **poprawny**, a moja propozycja „max” w R4 była błędna (zob. §3). Max z 6 zaszumionych SD jest dodatkowo obciążony w górę, czyli działa zachowawczo. |
| R4-02 | **USUNIĘTY** | §13.10 i §7.8(d) (QMNIST-50k, rozbieżność na pierwszej stronie). Test 9.1.5 obejmuje QMNIST, hash jest w `SHA256SUMS`. |
| R4-03 | **USUNIĘTY** | Wszystkie komórki są opisowe. Działa reguła listy komparatorów z nieliniowością wyjściową, a mechanizm wolno opisywać tylko w części eksploracyjnej. Nowa nieścisłość terminologiczna po D2 opisana jest w R5-01. |
| R4-04 | **USUNIĘTY** | §8: `rsync` z hashem po każdym biegu. Bieg jest ukończony dopiero po kopii lokalnej. |
| R4-05 | **USUNIĘTY** | Test 9.1.9 ma kryterium ustalone z góry. Przy tych samych porcjach 1 000 test sprawdza w praktyce poprawność indeksowania bufora, i to wystarcza. Test 9.1.6 jest powtarzany na każdej instancji. Generatory działają na CPU, ziarno to SHA-256 krotki. |
| R4-06 | **USUNIĘTY** | Jest operacyjna definicja błędu kodu (test, który nie przechodzi przed poprawką, a przechodzi po niej). Metryki walidacyjne trafiają do `zaslepione/`. |
| R4-07 | **USUNIĘTY** | Reguła działa po A i po B, w każdej warstwie, najwyżej 2 razy. Ujęto to w scenariuszu najgorszym. |
| R4-08 | **USUNIĘTY** | §7.6: NaN w warstwie 2 → argmax G₁. |
| R4-09 | **USUNIĘTY** | Kategoria NIESPÓJNY-ZGODNY-KIERUNEK i §13.11. Drobna uwaga o brzmieniu zdania: R5-05. |
| R4-10 | **USUNIĘTY** | §4.1, §9.1.10, §10.1. |
| R4-11 | **CZĘŚCIOWO** | 8 z 9 punktów poprawionych. Zostaje rozjazd liczby i opisu biegów BP w Etapie B (R5-02) oraz nieuaktualniona tabela pod-limitów w D1 (R5-06). |

---

## 3. Kierunek ograniczenia ρ (R4-01)

Oba cele mocy w §7.5 są testami IUT:
- (i) klasa P wymaga sup(0) dla każdego K;
- (ii) klasa R wymaga eq dla każdego K.

Moc IUT to prawdopodobieństwo, że m zdarzeń zajdzie jednocześnie. Przy dodatnio skorelowanych statystykach jest ono większe niż przy niezależnych. Niższe ρ daje więc niższą moc. Przyjęcie ρ ≤ 0.5 (gdy ρ̂ jest wyższe) zaniża moc, czyli zawyża N, a to kierunek zachowawczy.

Autor ma rację. Moja propozycja „max(ρ̂, 0.5)” w R4-01 była odwrotna, przepraszam za błąd.

Przeliczenie: MNIST, S = 0.30. F-MNIST: S = 1.2, σ = 0.5 (założenia). Wartości to moc klasy P / moc klasy R.

| m | ρ | σ_d MNIST | N = 30 | N = 40 | N = 50 | N = 60 |
|---|---|---|---|---|---|---|
| 6 | 0.0 | 0.45 | 0.86 / 0.34 | 0.95 / 0.74 | 0.99 / 0.92 | 1.00 / 0.98 |
| 6 | 0.25 | 0.45 | 0.86 / 0.36 | 0.95 / 0.75 | 0.99 / 0.92 | 1.00 / 0.98 |
| 6 | 0.5 | 0.45 | 0.86 / 0.45 | 0.95 / 0.79 | 0.99 / 0.94 | 1.00 / 0.98 |

Moc klasy R rośnie z ρ: 0.74 → 0.79 przy N = 40. Kierunek jest więc potwierdzony.

**Uwaga (bez wagi).** Dolne ograniczenie „nie mniej niż 0” nie jest zachowawcze w ścisłym sensie, bo ujemne ρ obniżyłoby moc jeszcze bardziej. Różnice `K-LGT − K-x` mają jednak wspólny składnik Acc(K-LGT), więc ρ < 0 jest mało prawdopodobne, a różnica mocy między ρ = 0 i 0.25 jest pomijalna.

---

## 4. D2: m = 6 w H1

### 4.1. Logika §7.3–7.4

Zmiana m nie wpływa na definicję warunków:
- IUT: sup i eq dla każdego K;
- test sumy: Holm na m wartościach p.

Przeliczyłem błąd I rodzaju dla m = 6 (40 000 powtórzeń, σ = 0.4, S = 0.3):

| Stan prawdziwy | Fałszywy werdykt | N = 30 | N = 60 | Wymaganie |
|---|---|---|---|---|
| Δ = 0 wobec jednego, 2S wobec reszty | klasa P | 0.0085 | 0.0086 | ≤ α'/2 = 0.0083 (MC ± 0.0005) ✓ |
| Δ = 0 wobec wszystkich | klasa G | 0.0067 | 0.0007 | ≤ 0.0083 ✓ |
| Δ = S wobec jednego, 0 wobec reszty | klasa R | 0.0077 | 0.0155 | ≤ α' = 0.0167 ✓ |

### 4.2. Moc

Parametry: MNIST S = 0.30, ρ = 0.5, oba zbiory. Wartości to moc klasy P / moc klasy R.

| σ_d | m = 5, N = 30 | m = 6, N = 30 | m = 5, N = 40 | m = 6, N = 40 | m = 6, N = 50 |
|---|---|---|---|---|---|
| 0.34 | 0.98 / 0.95 | 0.99 / 0.94 | 1.00 / 1.00 | 1.00 / 0.99 | 1.00 / 1.00 |
| 0.40 | 0.94 / 0.76 | 0.93 / 0.73 | 0.98 / 0.94 | 0.99 / 0.94 | 1.00 / 0.99 |
| 0.45 | 0.86 / 0.49 | 0.86 / 0.45 | 0.95 / 0.82 | 0.95 / 0.79 | 0.99 / 0.94 |

**Wnioski:**
- m = 6 kosztuje najwyżej ok. 0.04 mocy klasy R. Wiążący pozostaje cel (ii).
- Przy realistycznym σ_d = 1.29 × 0.34 ≈ 0.44 reguła wybierze najpewniej N = 50, przy σ_d ≈ 0.40 – N = 40.
- Budżet: 18 × 50 × 2 = 1 800 biegów w Etapie C mieści się w moim szacunku z R4 (R4-04). Nie widzę zagrożenia dla limitu 38 USD, ale rozstrzygnie to przeliczenie po mini-pilocie (§11.2).

### 4.3. Identyfikacja

Dodanie `K-LLT` i `K-GGT` jest trafne. H1 = P wyklucza teraz wyjaśnienie „wystarczy tanh na wyjściu”, a zdanie dopuszczone w §7.4 wynika wprost z testu. Wprowadza to jednak nieścisłość terminologiczną (R5-01).

---

## 5. D3: E9 i E10

### 5.1. Furtki do części konfirmacyjnej

Nie znalazłem furtek:
- E9 korzysta z konfiguracji z Etapu B (bez sprzężenia zwrotnego).
- Metryki walidacyjne E9 są zaślepione, a test oceniany jest w tym samym kroku §5.6.
- Cięcia budżetowe uderzają najpierw w E9, nie w N (§11.3).
- E10 jest lokalne, ma osobne ziarna i nie dotyka MNIST/F-MNIST.
- Biegi E9 przechodzą przez wspólny kod, więc poprawka błędu w ścieżce małych szerokości uruchomi regułę pokrycia także dla ramion głównych. To działa zachowawczo.

### 5.2. Reguła wyboru rysunku E10

Reguła w zasadzie chroni przed wybieraniem korzystnych przykładów. Z góry ustalone są:
- budżet (w = 16),
- wszystkie ramiona,
- kryterium wyboru ziarna (mediana walidacji),
- siatka i zakres,
- pełna siatka w materiałach uzupełniających.

Zostaje jedna luka: przy 10 ziarnach mediana nie wskazuje jednego ziarna (R5-03).

### 5.3. Liczby

**E9, wymiary K.** Liczyłem dopasowanie per warstwa: blok 1 = h² + (786 + w)h + w, blok 2 = h² + (2w + 2)h + w.

| w | P | K | Różnica | Czy h najbliższe? |
|---|---|---|---|---|
| 32 | 25 120 + 1 056 = **26 176** ✓ | 25 472 + 1 059 = **26 531** ✓ | +1.36% ✓ | h₁ = 30 (29 daje −2.1%), h₂ = 13 ✓ |
| 64 | 50 240 + 4 160 = **54 400** ✓ | 49 839 + 4 120 = **53 959** ✓ | −0.81% ✓ | h₁ = 55 (56 daje +1.1%), h₂ = 26 (27: +3.4%) ✓ |
| 128 | 100 480 + 16 512 = **116 992** ✓ | 100 415 + 16 611 = **117 026** ✓ | +0.03% ✓ | h₁ = 99, h₂ = 53 (52: −1.6%, 53: +0.6%) ✓ |

**E10, u.** P: wejście 4 → w → w, czyli w² + 6w = 112, 352, 1 216, 4 480. K: 4 → u → u → u ‖ u → u → u → u, czyli 5u² + 10u:

| w | P | u | K | Sąsiad |
|---|---|---|---|---|
| 8 | 112 | 4 | 120 | u = 3: 75 |
| 16 | 352 | 7 | 315 | u = 8: 400 (dalej) |
| 32 | 1 216 | 15 | 1 275 | u = 14: 1 120 (dalej) |
| 64 | 4 480 | 29 | 4 495 | u = 28: 4 200 |

Wszystkie u = 4, 7, 15, 29 są poprawne.

**§11.2.**
- Poprawne: mini-pilot 160, A 576 + 64, B 324 + ≤ 432, C 1 080 / 2 160, E2 40, E9 360.
- E10: 7 × 4 × (8 + 10) = 504, zgodne pod warunkiem, że strojenie odbywa się per (ramię, budżet). Zob. R5-03.
- Brakuje biegów Etapu B dla BP (R5-02).

---

## 6. Spójność dokumentu

Sprawdzone i poprawne:
- §13 ma punkty 1–13, a odwołania §13.10 i §13.11 (z §12.4 i O4) trafiają;
- §12.4 odpowiada O4;
- D1 pkt 2 odsyła do §11.3;
- liczby z §3.1 bez zmian;
- §9.2 obejmuje 18 ramion;
- §7.5 ma m = 6.

Niespójności: R5-02, R5-06.

---

## 7. Nowe zarzuty

| ID | Waga | Sekcja | Problem i poprawka |
|---|---|---|---|
| R5-01 | DROBNY | §1 (H1), §7.4 (macierz, H1 = P), §7.7 (E9) | **„Bloki jednorodne” obejmują teraz `K-LLT` i `K-GGT`, które jednorodne nie są** (dwa rodzaje nieliniowości). Zdania z macierzy są wiążące, więc np. przy H1 = G z przegraną od `K-LLT` raport stwierdzi „LGT gorszy od bloku jednorodnego”, co jest fałszywym opisem. **Poprawka:** nazwać grupę H1 „blokami referencyjnymi” (4 jednorodne + 2 jednorodne zakończone tanh) i podmienić termin we wszystkich 16 komórkach, w §1, w E9 („najlepsze ramię referencyjne”) i w szablonie. |
| R5-02 | DROBNY | §5.4, §5.7, §11.2 | **BP:** §5.4 mówi „E2 używa konfiguracji BP z Etapu B”, ale Etap B (procedura i liczby w §11.2) obejmuje tylko 18 ramion FF. Brakuje 2 × 9 × 2 = 36 biegów (plus reguła brzegowa). **Poprawka:** dopisać BP do Etapu B (procedura jak FF, 2D) i do §11.2 albo wrócić do wyboru z Etapu A. |
| R5-03 | DROBNY | §7.7 (reguła rysunku E10), §5.7 (E10) | (a) Przy 10 ziarnach „ziarno o medianie” jest nieokreślone, bo mediana leży między 5. i 6. wartością. „Remis → niższe ziarno” tego nie rozstrzyga. **Poprawka:** „5. wartość w porządku rosnącym (dolna mediana); przy równych dokładnościach niższy numer ziarna”. (b) Strojenie E10 na ziarnie 50001, które jest też jednym z 10 ziaren ewaluacyjnych, w przeciwieństwie do rozłącznych pul w badaniu głównym. Dla eksploracji jest to dopuszczalne, ale lepiej użyć ziarna 50000. (c) Doprecyzować, że wybór punktu strojenia odbywa się per (ramię, budżet), zgodnie z liczbą 504. |
| R5-04 | DROBNY | §7.7 (E9) | „Najlepsze ramię jednorodne tej samej topologii” nie ma kryterium wyboru. Wybór po średniej testowej to selekcja maksimum: działa przeciw przewidywaniu autora, ale jest niejasny. Do tego, przy każdej szerokości, może wygrywać inne ramię. **Poprawka:** ustalić z góry, np. najlepsze ramię referencyjne według średniej walidacyjnej (z `zaslepione/` po kroku 2) dla każdego w osobno, i zawsze raportować różnice względem każdego ramienia. |
| R5-05 | DROBNY | §7.4 | Zdanie NIESPÓJNY-ZGODNY-KIERUNEK: „na zbiorze X różnica przekracza poziom istotności statystycznej” jest niejasne. Klasa P może oznaczać PRZEWAGĘ albo PRZEWAGĘ<S. **Poprawka:** „na zbiorze X: [kategoria zbiorowa]; na zbiorze Y: różnica istotna statystycznie w tym samym kierunku, praktycznie pomijalna (< S, równoważność wykazana)”. |
| R5-06 | DROBNY | `DECYZJE_AUTORA.md` D1 | Tabela pod-limitów w D1 nadal ma wiersz „Etap C + E2” (bez E9) i brzmi jak twarde limity, a §11.1 nazywa je planistycznymi. **Poprawka:** zsynchronizować D1 z §11.1. |

Nie znalazłem nowych problemów KRYTYCZNYCH ani POWAŻNYCH z realnym scenariuszem błędu.

---

## 8. Czy protokół jest gotowy do zamrożenia?

**Tak, po wprowadzeniu poprawek R5-01 do R5-06, bez kolejnej rundy recenzji.**

Wszystkie poprawki są tekstowe albo porządkowe. Żadna nie zmienia logiki decyzyjnej, kontroli błędu ani mocy.

Kolejność dalszych kroków pozostaje taka, jak w §9 i §10:
1. kod i testy §9.1–9.2;
2. mini-pilot (8 ramion), wpisanie N, S, σ_d, ρ̂, E_max i budżetu;
3. tag `ff-od-zera-prereg`;
4. zewnętrzny znacznik czasu przed Etapem A.
