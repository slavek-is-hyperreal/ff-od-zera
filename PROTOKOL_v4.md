# Protokół badania: heterogeniczność i kolejność nieliniowości w sieciach Forward-Forward przy stałym budżecie parametrów

**Wersja:** v4 (kandydat do zamrożenia, NIE zamrożony)
**Data:** 23.10.2026
**Autor:** Sławomir Majcher
**Katalog:** `/vectorlegis_ssd_pool/uczelnia/eksperyment_ff_od_zera/`
**Dokumenty powiązane:**
- `00_AUDYT_BLEDOW.md` – błędy poprzednich eksperymentów (identyfikatory S*, L*, K*, G*)
- `recenzje/R1.md` – recenzja wersji v1 (identyfikatory R1-*)
- `odpowiedzi/O1.md` – odpowiedź na R1
- `recenzje/R2.md`, `odpowiedzi/O2.md` – runda 2
- `recenzje/R3.md`, `odpowiedzi/O3.md` – runda 3
- `DECYZJE_AUTORA.md` – decyzja D1 (sprzęt Vast.ai, limit finansowy)

Ten protokół zastępuje w całości protokoły z `eksperyment_sca_ff`, `eksperyment_laminar_ff` i `eksperyment_kontrola_wymiaru_ff`. Wyniki tamtych eksperymentów służą wyłącznie jako dane pilotowe (analiza mocy, szacunek czasu). Nie są cytowane jako wyniki.

---

## 1. Pytanie badawcze i zakres

**Pytanie główne.** W sieciach MLP uczonych algorytmem Forward-Forward (FF), przy równym budżecie parametrów na warstwę FF, równej głębokości i równym budżecie strojenia hiperparametrów:

1. **H1.** Czy blok trzech podwarstw z nieliniowościami LeakyReLU → GELU → tanh („blok LGT”) daje inną dokładność niż **każdy** blok jednorodny o tej samej topologii (ReLU, LeakyReLU, GELU, tanh we wszystkich trzech podwarstwach)?
2. **H2.** Czy blok LGT daje inną dokładność niż **każda** inna kolejność tych samych trzech nieliniowości?
3. **H3.** Czy w płaskiej sieci FF (jedna warstwa liniowa + nieliniowość na warstwę FF) warstwa z równoległym bankiem LeakyReLU | GELU | tanh, z dobrocią liczoną jako suma Σa² po wszystkich jednostkach, daje inną dokładność niż warstwa z **każdą** z czterech pojedynczych nieliniowości?

**Zakres twierdzeń.** H1 i H2 dotyczą tego konkretnego bloku (L→G→T, bez normalizacji wewnątrz bloku), a nie „różnorodności nieliniowości” w ogóle (por. R1-02). H3 dotyczy banku z dobrocią Σa² po wszystkich jednostkach (por. R1-12).

**Czego badanie nie rozstrzyga:**
- zachowania na danych innych niż MNIST i Fashion-MNIST,
- architektur splotowych,
- innych schematów FF,
- trafności analogii z budową kory mózgowej (była motywacją, nie jest przedmiotem testu; por. L6).

**Status:** konfirmacyjne dla H1–H3 (§2, §7.3–7.4), eksploracyjne dla wszystkich pozostałych analiz (§7.7).

---

## 2. Hipotezy konfirmacyjne

Dla każdej pary (ramię badane B, komparator K) i każdego zbioru danych: Δ_K = Acc_B − Acc_K (p.p., zbiór testowy), parowane po ziarnie (§4.3).

| ID | Ramię badane | Grupa komparatorów (każdy osobno) | Przewidywanie autora |
|---|---|---|---|
| **H1** | `K-LGT` | `K-R`, `K-L`, `K-G`, `K-T` (m = 4) | przewaga |
| **H2** | `K-LGT` | `K-LTG`, `K-GLT`, `K-GTL`, `K-TLG`, `K-TGL` (m = 5) | przewaga |
| **H3** | `P-BANK` | `P-R`, `P-L`, `P-G`, `P-T` (m = 4) | przewaga |

Werdykty są definiowane względem **całej grupy** (testy przecięcia-sumy i sumy-przecięcia, §7.3), a nie względem jednego wybranego komparatora. Dzięki temu nie ma etapu selekcji „najlepszego” konkurenta, który mógłby zawyżać przewagę (por. R1-01).

Przewidywanie autora jest kierunkowe. Testy są dwustronne, a każdy możliwy wynik ma z góry zapisaną interpretację (§7.4). Wynik przeciwny przewidywaniu jest raportowany na równi ze zgodnym (por. K4, K5).

---

## 3. Ramiona eksperymentu

### 3.1. Topologie

**Płaska (P).** Dwie warstwy FF, każda = Linear + nieliniowość.
- warstwa 1: 784 → 250 (196 250 parametrów)
- warstwa 2: 250 → 250 (62 750 parametrów)
- razem **259 000**

**Blokowa (K).** Dwie warstwy FF, każda = blok trzech podwarstw (każda podwarstwa = Linear + nieliniowość; brak złożeń liniowo-liniowych, por. L1).
- Wewnątrz bloku gradient straty FF propaguje się przez trzy podwarstwy.
- Między warstwami FF gradient nie przechodzi (`detach`).
- blok 1: 784 → 163 → 163 → 250 (195 687 parametrów)
- blok 2: 250 → 103 → 103 → 250 (62 565 parametrów)
- razem **258 252** (−0.29% względem P; każda warstwa FF osobno −0.29%)

### 3.2. Nieliniowości
- **R** = ReLU
- **L** = LeakyReLU(ujemne nachylenie 0.5)
- **G** = GELU (`approximate='none'`)
- **T** = tanh
- **BANK** = ciągłe grupy jednostek L | G | T o licznościach ⌈n/3⌉, ⌈(n−1)/3⌉, ⌊n/3⌋. Dla n = 250: 84/83/83; dla n = 163: 55/54/54; dla n = 103: 35/34/34.

Wyjściem warstwy FF jest wyjście ostatniej podwarstwy. Z niego liczona jest dobroć.

### 3.3. Ramiona

**Etapy A i C: 16 ramion FF.**

| Ramię | Topologia | Nieliniowości (podwarstwa 1→2→3) | Rola |
|---|---|---|---|
| `P-R`, `P-L`, `P-G`, `P-T` | P | jedna | komparatory H3 |
| `P-BANK` | P | BANK | badane H3 |
| `K-R`, `K-L`, `K-G`, `K-T` | K | X→X→X | komparatory H1 |
| `K-LGT` | K | L→G→T | badane H1, H2 |
| `K-LTG`, `K-GLT`, `K-GTL`, `K-TLG`, `K-TGL` | K | permutacje | komparatory H2 |
| `K-BANK` | K | BANK→BANK→BANK | eksploracyjne (E3) |

**Analiza poboczna E2 (osobno, N = 10, §7.7): 2 ramiona BP.**
- `BP-P-R`: topologia P + odczyt 250→10 (261 510 parametrów)
- `BP-K-LGT`: topologia K + odczyt 250→10 (260 762 parametrów)
- Uczone end-to-end entropią krzyżową, na obrazach bez nałożonej etykiety (por. K8).

`P-R` to „FF w stylu Hintona” w małej skali, a nie odtworzenie Hintona (2022). Tamta praca używa 4 × 2000 ReLU, 60 epok, innych negatywów i inferencji bez pierwszej warstwy; 1.36% błędu (por. R1-15).

Ramię „ReLU D = 1000” z pilota zostało usunięte, bo nie dotyczy żadnej hipotezy.

---

## 4. Dane

### 4.1. Zbiory
- **MNIST** (LeCun i in., 1998) i **Fashion-MNIST** (Xiao i in., 2017): po 60 000 obrazów treningowych i 10 000 testowych.
- Oba zbiory są konfirmacyjne.
- Przetwarzanie: piksele /255 → [0, 1], bez augmentacji, spłaszczenie do 784.

### 4.2. Podziały
- **Test (10 000):** używany wyłącznie w kroku ewaluacji końcowej (§5.6), który obejmuje modele Etapu C i modele E2. Runner w trybach treningowych nie ma ścieżki kodu do zbioru testowego (test 9.1.5).
- **Trening 60 000:** dla każdego ziarna losowy podział 50 000 (trening) / 10 000 (walidacja). Wnioskowanie obejmuje więc także zmienność podziału danych (por. K13).
- **Wczesne zatrzymanie:** stały (dla danego ziarna) podzbiór 5 000 przykładów walidacji (strumień `split`).
- **Wybór konfiguracji w Etapie B:** pełne 10 000 walidacji.

### 4.3. Ziarna, strumienie, parowanie

Ziarno s wyznacza niezależne generatory `torch.Generator`, seedowane deterministycznie z krotki:
- `split`: (s, 1) – podział i podzbiór 5k
- `order`: (s, 2, warstwa, epoka) – kolejność batchy
- `neg`: (s, 3, warstwa, epoka) – etykiety negatywne
- `init`: (s, 4) – inicjalizacja wag

Strumienie `split`, `order` i `neg` są identyczne dla wszystkich ramion przy tym samym ziarnie, niezależnie od tego, kiedy zatrzymała się warstwa 1 (por. R1-16). To jest podstawą parowania.

**Pule ziaren** (rozłączne między sobą i z pilotem; por. K18, L7):

| Pula | Ziarna |
|---|---|
| mini-pilot (§9.3) | 40001–40010 |
| Etap A | 30001 |
| Etap B | 20001–20003 |
| Etap C | 10001 … 10000+N (N z §7.5) |
| E2 (BP) | 10001–10010 (te same co pierwsze 10 z Etapu C, parowanie z FF) |

---

## 5. Procedura

### 5.1. Forward-Forward
- **Etykieta:** piksele 0–9 zerowane, piksel y ustawiany na 1.0. Operacja wykonywana po skalowaniu do [0, 1].
- **Negatywy:** ten sam obraz z etykietą y' ~ U({0..9} \ {y}), losowaną na nowo w każdym batchu (strumień `neg`).
- **Dobroć:** G = Σⱼ aⱼ² po jednostkach wyjścia warstwy FF.
- **Strata:** L = mean softplus(θ − G_pos) + mean softplus(G_neg − θ).
- **Próg:** θ = c · 250, c wspólne dla obu warstw FF danego ramienia (ograniczenie przestrzeni strojenia, §13).
- **Wejście warstwy 2:** a / (‖a‖₂ + 1e-8), wyjście warstwy 1 po normalizacji L2.
- **Uczenie zachłanne:** warstwa 1 do zatrzymania (§5.2), zamrożenie, potem warstwa 2.
- **Buforowanie:** podczas uczenia warstwy 2 wyjścia zamrożonej warstwy 1 są buforowane dla wszystkich 10 wariantów etykiety. Nie zmienia to wyników (test 9.1.9).
- **Optymalizacja:** AdamW, batch 128, `drop_last=True`, stały lr, przycinanie normy gradientu do 10 (wszystkie ramiona; por. R1-10).
- **Klasyfikacja:** ŷ = argmax_c Σₗ Gₗ(x z etykietą c). Odjęcie stałych progów nie zmienia argmax, więc θ nie wchodzi do inferencji (por. R1-14).

### 5.2. Epoki i zatrzymanie
- Limit E_max epok na warstwę FF, wczesne zatrzymanie z cierpliwością 5 epok na dokładności podzbioru 5k:
  - warstwa 1: klasyfikacja z G₁,
  - warstwa 2: klasyfikacja z G₁ + G₂.
- Przywracane są wagi z najlepszej epoki.
- **Wartość E_max** ustalana przed zamrożeniem z mini-pilota (§9.3): E_max = max(30, ⌈1.5 × 90. percentyla epoki najlepszej w mini-pilocie⌉), nie więcej niż 80.
- **Reguła po Etapie B** (por. R1-04): jeśli dla któregokolwiek ramienia > 25% biegów Etapu B ma najlepszą epokę ≥ E_max − 2, E_max dla **wszystkich** ramion w Etapie C rośnie o 50%. Decyzja zapisywana w `ETAP_B_WYBOR.json`.

### 5.3. Etap A – strojenie z równym budżetem
- **Przestrzeń:**
  - lr ∈ [1e-4, 1e-2] (log)
  - wd ∈ [1e-6, 1e-2] (log)
  - c ∈ [0.01, 1] (log), czyli θ ∈ [2.5, 250] (por. R1-03)
- **Punkty:** 16 punktów scrambled Sobol (ziarno 0), identycznych dla wszystkich 16 ramion FF.
- 1 bieg na punkt (ziarno 30001), kryterium: dokładność na pełnej walidacji. Osobno dla każdego zbioru.

### 5.4. Etap B – wybór konfiguracji
- Dla każdego ramienia i zbioru: 3 najlepsze punkty z Etapu A × 3 ziarna (20001–20003).
- Konfiguracja = punkt o najwyższej średniej dokładności walidacyjnej (remis → niższe c).
- **Reguła brzegowa** (symetryczna; por. R1-03, R2-07):
  1. Oś „brzegowa” to oś, na której konfiguracja **któregokolwiek** ramienia (w danym zbiorze) leży w skrajnych 10% zakresu (skala log).
  2. Dla każdej osi brzegowej (najwyżej 2 osie; przy 3 – dwie o największej liczbie ramion na brzegu, remis → kolejność lr, wd, c) **każde** ramię dostaje 2 dodatkowe punkty: własna konfiguracja przesunięta o ±25% szerokości tej osi (log), pozostałe współrzędne bez zmian. Maksymalnie 4 punkty na ramię.
  3. Punkty mogą wychodzić poza zakres z §5.3, z jednym wyjątkiem: c jest obcinane do 1. Jeśli obcięcie daje punkt już oceniony, punkt jest pomijany, a liczba ocenionych punktów raportowana per ramię.
  4. Punkty oceniane tak samo (3 ziarna). Wybór konfiguracji ponownie według reguły wyżej, spośród wszystkich ocenionych punktów ramienia. Reguła stosowana jednokrotnie.
  5. Budżet (§11) zakłada, że reguła się uruchomi.
- Wynik zapisywany do `ETAP_B_WYBOR.json`, commit i hash **przed** Etapem C. Zawiera konfiguracje, E_max i decyzje z reguł.

### 5.5. Etap C – trening konfirmacyjny
- 16 ramion × N ziaren × 2 zbiory.
- Zapisywane są:
  - wagi z najlepszej epoki,
  - krzywe uczenia,
  - metryki walidacyjne.
- **Bez ewaluacji testowej** (por. R1-08).

### 5.6. Ewaluacja końcowa (zaślepiona do końca treningu)
1. Po zakończeniu **wszystkich** biegów Etapu C: commit pliku z hashami SHA-256 wszystkich checkpointów.
2. Jedno polecenie ocenia wszystkie checkpointy na zbiorze testowym i zapisuje predykcje per przykład.
3. Uruchamiany jest skrypt analizy zamrożony w §10.
4. **Powtórki po błędzie kodu (przed krokiem 2)** (por. R2-04):
   - zakres powtórki = **wszystkie** ramiona, ziarna i zbiory etapu, których wykonanie przechodzi przez zmieniony kod;
   - zakres wyznacza automatycznie skrypt pokrycia (lista funkcji zmienionych w diffie → biegi, które je wywołują); w razie wątpliwości – wszystkie biegi etapu;
   - diff, lista biegów i powód są zapisywane w `ODSTEPSTWA.md` i commitowane **przed** ponownym uruchomieniem;
   - checkpointy biegów pierwotnych są zachowywane i oceniane w kroku 2 razem z pozostałymi, a ich wyniki raportowane w §7.8(c).
5. Powtórki po kroku 2 są dopuszczalne tylko jako analiza wrażliwości, a wynik pierwotny pozostaje wynikiem głównym.

### 5.7. Specyfikacja implementacyjna (por. R1-11)

| Element | Wartość |
|---|---|
| Inicjalizacja | domyślna PyTorch `nn.Linear` (wagi i bias: Kaiming-uniform a = √5 / U(±1/√fan_in)), jednakowa dla wszystkich ramion |
| Bias | tak, w każdej warstwie liniowej |
| dtype | float32 |
| AdamW | betas (0.9, 0.999), eps 1e-8 |
| Eps normalizacji | 1e-8 |
| BANK | podział jak §3.2; dobroć = suma po wszystkich jednostkach |
| BP | entropia krzyżowa, AdamW, batch 128, przycinanie 10; wczesne zatrzymanie jak §5.2 (dokładność walidacyjna), przywrócenie najlepszych wag; strojenie: te same 16 punktów (lr, wd), ziarno 30001 |
| Determinizm | `torch.use_deterministic_algorithms(True)`; na GPU dodatkowo `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Sprzęt | GPU wynajęte w Vast.ai (decyzja D1); cały zbiór danych w pamięci GPU; kilka biegów równolegle na jednej karcie, każdy w osobnym procesie |
| Wątki CPU | `torch.set_num_threads(1)` na proces, `set_num_interop_threads(1)` |
| Środowisko | jeden obraz Dockera (identyfikowany hashem) na cały etap; wersje zapisywane w każdym JSON-ie |
| Wersje | Python, PyTorch, CUDA, sterownik i hash commitu zapisywane w każdym JSON-ie biegu |

Inicjalizacja domyślna nie jest optymalna dla wszystkich nieliniowości. Wyniki są warunkowe względem niej (§13).

---

## 6. Metryki

### 6.1. Główna
Dokładność testowa (%), ewaluacja z §5.6. Predykcje per przykład są zapisywane.

### 6.2. Drugorzędne (opisowe, §7.7)
Liczone na pozytywach walidacji, per warstwa FF:
- **Jednostki nieaktywne:** sd(aⱼ) < 0.01 · P90ₖ sd(aₖ), plus pełny rozkład sd (por. R1-18).
- **Participation ratio:** PR = (Σλ)² / Σλ², λ = wartości własne macierzy kowariancji wyjść.
- **Udział grup BANK** (L/G/T) w G_pos, G_neg i normie gradientu straty, per epoka (E8).
- **Przebieg uczenia:** krzywe strat i dokładności walidacyjnej, epoka najlepsza, odsetek biegów z aktywnym limitem epok, odsetek biegów z NaN.
- **Czas ściany:** opisowo (por. L4, K15).

---

## 7. Analiza statystyczna

### 7.1. Estymand
Estymandem głównym jest **różnica dokładności na oficjalnym zbiorze testowym, uśredniona po losowości treningu** (inicjalizacja, kolejność, negatywy, podział 50k/10k). Wnioski są warunkowe względem tego zbioru testowego (por. R1-07).

Analiza wrażliwości §7.8 ocenia, czy werdykt utrzymuje się po uwzględnieniu próbkowania przykładów testowych.

### 7.2. Test elementarny
- Dla pary (B, K) i zbioru: dᵢ = Acc_B(sᵢ) − Acc_K(sᵢ), i = 1..N.
- Przedział ufności t-Studenta dla średniej d.
- Bez wstępnego testu normalności i bez warunkowego wyboru testu (por. K11).
- α = 0.05, korekta Bonferroniego na 3 hipotezy: **α' = 0.0167**.

### 7.3. Werdykt hipotezy na jednym zbiorze

Oznaczenia:
- CI_K = dwustronny (1 − α') PU dla Δ_K
- CI2_K = (1 − 2α') PU dla Δ_K (TOST)
- S = SESOI danego zbioru (§7.5)

**Warunki elementarne** (dla grupy komparatorów K₁…K_m):

| Warunek | Definicja | Typ testu |
|---|---|---|
| **sup(S)** | dla każdego K: dolna granica CI_K > S | IUT (bez korekty) |
| **sup(0)** | dla każdego K: dolna granica CI_K > 0 | IUT |
| **eq** | dla każdego K: CI2_K ⊂ (−S, S) | IUT (TOST) |
| **inf(S)** | jednostronne p_K dla H₀: Δ_K ≥ −S; procedura Holma na m wartościach p na poziomie α'/2; odrzucona co najmniej jedna H₀ | test sumy |
| **inf(0)** | jak wyżej, dla H₀: Δ_K ≥ 0 | test sumy |

Poziom α'/2 w inf(·) odpowiada poziomowi jednej strony dwustronnego CI_K używanego w sup(·), więc oba kierunki dostają jednakowe α (por. R2-06).

**Kategorie i kolejność przypisania** (pierwsza spełniona jest werdyktem):

| # | Kategoria | Warunek | Klasa w macierzy §7.4 |
|---|---|---|---|
| 1 | **GORSZOŚĆ** | inf(S) | G |
| 2 | **PRZEWAGA** | sup(S) | P |
| 3 | **RÓWNOWAŻNOŚĆ+** | eq ∧ (sup(0) ∨ inf(0)): różnica wykrywalna, ale praktycznie pomijalna (< S) | R |
| 4 | **GORSZOŚĆ<S** | inf(0) | G |
| 5 | **PRZEWAGA<S** | sup(0) | P |
| 6 | **RÓWNOWAŻNOŚĆ** | eq | R |
| 7 | **NIEROZSTRZYGNIĘTY** | – | N |

Uwagi:
- sup(S) i inf(S) wykluczają eq (granica CI2 leży wewnątrz CI), więc kategorie 1–2 nie kolidują z 3.
- sup(·) i inf(·) wymagają przeciwnych znaków granic, więc nie zachodzą jednocześnie (sprawdzone w R2).
- Kategorie 4–5 oznaczają różnicę istotną statystycznie, mniejszą niż S **bez** wykazanej równoważności. Jest to różnica, której wielkość pozostaje niepewna: może być praktycznie istotna albo nie.

**Brak przewagi nad grupą, ale istotne różnice z poszczególnymi członkami** (np. LGT > K-R, LGT ≈ K-G) dają werdykt NIEROZSTRZYGNIĘTY dla hipotezy. Kontrasty pojedyncze raportowane są w E3 jako eksploracyjne.

### 7.4. Łączenie zbiorów i interpretacja

**Werdykt hipotezy** (łączenie po klasach z §7.3; IUT po zbiorach, Berger i Hsu, 1996):
- **ta sama klasa** (P, R lub G) na obu zbiorach → ta klasa. Kategoria łączna to słabsza z dwóch kategorii zbiorowych:
  - P: PRZEWAGA tylko, gdy PRZEWAGA na obu, inaczej PRZEWAGA<S;
  - G: analogicznie;
  - R: RÓWNOWAŻNOŚĆ+, jeśli choć na jednym zbiorze wystąpiło RÓWNOWAŻNOŚĆ+, inaczej RÓWNOWAŻNOŚĆ.
  - W raporcie zawsze obie kategorie zbiorowe.
- **różne klasy:**
  - **NIESPÓJNY-KIERUNEK**: P na jednym, G na drugim zbiorze;
  - **NIESPÓJNY-CZĘŚCIOWY**: pozostałe kombinacje (np. P i N, R i N, P i R). Klasa w macierzy: N.

**Tabela interpretacji H1 × H2.** Wiersze: klasa H1, kolumny: klasa H2. Oznaczenia:
- „P” = PRZEWAGA lub PRZEWAGA<S
- „R” = RÓWNOWAŻNOŚĆ lub RÓWNOWAŻNOŚĆ+
- „G” = GORSZOŚĆ lub GORSZOŚĆ<S
- „N” = NIEROZSTRZYGNIĘTY lub NIESPÓJNY-*

**Obowiązkowe dopiski do zdania z komórki:**
- przy kategorii z „<S”: „różnica istotna statystycznie, mniejsza niż przyjęty próg praktycznej istotności S; jej praktyczne znaczenie nie zostało wykazane”;
- przy RÓWNOWAŻNOŚCI+: „wykryto różnicę statystycznie istotną, ale praktycznie pomijalną (mniejszą niż S)”, z kierunkiem.

| H1 \ H2 | P | R | G | N |
|---|---|---|---|---|
| **P** | Blok LGT lepszy od każdej monokultury i każdej innej kolejności: efekt specyficzny dla kolejności L→G→T. | LGT lepszy od każdej monokultury i równoważny każdej innej kolejności: liczy się zestaw nieliniowości w bloku, nie ich kolejność. | LGT lepszy od monokultur, ale istnieje lepsza kolejność: różnorodność pomaga, LGT nie jest najlepszą kolejnością. | LGT lepszy od monokultur; rola kolejności nierozstrzygnięta. |
| **R** | LGT równoważny monokulturom i lepszy od pozostałych kolejności: inne kolejności tracą względem monokultur, a LGT nie; kolejność ma znaczenie, ale LGT nie przewyższa najlepszej monokultury (sprawdzić E7: czy dotyczy permutacji bez tanh na wyjściu). | Brak efektu: LGT równoważny monokulturom i innym kolejnościom. | LGT równoważny monokulturom, istnieje lepsza kolejność: korzyść możliwa dla innej kolejności (E3, E7). | LGT nie różni się od monokultur w zakresie S; kolejność nierozstrzygnięta. |
| **G** | LGT gorszy od pewnej monokultury, lepszy od innych kolejności: różnorodność nie pomaga w porównaniu z najlepszą monokulturą. | LGT gorszy od pewnej monokultury, kolejność obojętna: mieszanie nieliniowości w bloku szkodzi. | LGT gorszy od pewnej monokultury i pewnej kolejności. | LGT gorszy od pewnej monokultury; kolejność nierozstrzygnięta. |
| **N** | LGT lepszy od innych kolejności; względem monokultur nierozstrzygnięty. | Kolejność obojętna; względem monokultur nierozstrzygnięty. | Istnieje lepsza kolejność; względem monokultur nierozstrzygnięty. | Badanie nie rozstrzyga H1 ani H2. |

Każda komórka z „P” lub „G” podaje w raporcie, która z wersji (pełna czy „<S”) zaszła.

**Tabela H3:**

| Werdykt | Interpretacja |
|---|---|
| P | Bank (Σa² po wszystkich jednostkach) lepszy od każdej pojedynczej nieliniowości w topologii płaskiej. |
| R | Bank równoważny każdej pojedynczej nieliniowości (w zakresie S). |
| G | Istnieje pojedyncza nieliniowość lepsza od banku. |
| N | Nierozstrzygnięty. |

Każda interpretacja jest ograniczona zakresem z §1 i §13.

### 7.5. SESOI, liczebność próby i moc

**SESOI** (por. R1-13, R2-05): 10% błędu walidacyjnego ramienia tej samej topologii co ramię badane, zmierzonego w mini-pilocie (§9.3), osobno per zbiór, zaokrąglone do 0.05 p.p. w górę:
- H1, H2: S = 0.10 × błąd `K-LGT`;
- H3: S = 0.10 × błąd `P-R`.

Konfiguracja mini-pilota nie jest strojona, więc błąd bazowy może być zawyżony, a S nieco szerszy niż 10% błędu strojonego modelu. Wartości wpisywane są do protokołu przed zamrożeniem.

**Moc** jest liczona symulacją całego potoku §7.3–7.4 (IUT po grupie, oba zbiory), w najgorszym przypadku (por. R3-02):
- m = 5;
- S = min(S dla H1/H2, S dla H3) danego zbioru;
- σ_d per zbiór = max(1.29 × SD różnic `K-LGT` − `P-R` po ziarnach mini-pilota, 0.5 · S). Czynnik 1.29 to górna 80% granica ufności SD przy 9 stopniach swobody; chroni przed zaniżeniem N przez niedoszacowane σ_d (por. R3-01);
- korelacja różnic w grupie ρ = 0.5.

Do protokołu wpisywane są również moce przy σ_d = 1.47 × SD (górna granica 90%), jako informacja o odporności.

**Cele mocy** (por. R2-01):
- **(i) klasa P:** P(werdykt łączny w klasie P) ≥ 0.80, gdy prawdziwe Δ = S względem jednego komparatora i 2S względem pozostałych.
- **(ii) klasa R:** P(werdykt łączny w klasie R) ≥ 0.80, gdy prawdziwe Δ = 0 względem wszystkich.
- Kategoria PRZEWAGA (> S) jest **drugorzędna**: dla niej nie ma celu mocy. Jej moc w scenariuszu Δ = 2S / 3S jest raportowana.

**Reguła N:** N = najmniejsze z {30, 40, 50, 60} spełniające (i) i (ii). Jeśli N = 60 ich nie spełnia, N = 60, a uzyskane moce są wpisywane do protokołu przed zamrożeniem. Wartości N, S, σ_d i moce są dopisywane do tej sekcji.

**Brak analiz pośrednich** (por. K3). Monitor postępu pokazuje tylko liczbę ukończonych biegów i błędy, bez żadnej metryki dokładności (§8).

### 7.6. NaN i niestabilność (por. R1-10)
- Bieg z NaN w stracie zatrzymuje się. Wynikiem jest model z najlepszej dotychczasowej epoki. Gdy takiej nie ma, wyjście jest losowe ze strumienia `init`, co daje ok. 10%.
- Bieg pozostaje w analizie.
- Jeśli w danej hipotezie i zbiorze wystąpi choć jeden taki bieg, werdykt liczony jest dodatkowo testem Wilcoxona (te same progi, PU Hodgesa–Lehmanna). Przy niezgodności kategorii werdykt brzmi **NIEROZSTRZYGNIĘTY (niestabilność)**.

### 7.7. Analizy eksploracyjne
Bez wniosków konfirmacyjnych, bez korekty, zawsze opisywane jako eksploracyjne.

| ID | Analiza |
|---|---|
| E1 | Głębokość: `K-x` vs `P-x` dla x ∈ {R, L, G, T, BANK} |
| E2 | Luka do BP (N = 10): `BP-P-R` − `P-R`, `BP-K-LGT` − `K-LGT`. Zastrzeżenie: BP strojone w 2D (16 punktów), FF w 3D. |
| E3 | Wszystkie kontrasty pojedyncze w grupach H1–H3 oraz `K-BANK` vs `K-LGT` i vs monokultury K |
| E4 | Metryki drugorzędne (§6.2) |
| E5 | Krajobraz strojenia z Etapu A, z zaznaczeniem zakresu c nieosiągalnego dla wyjść tanh (obowiązkowy) |
| E6 | Inferencja w wariancie Hintona: dobroć tylko z warstwy 2 |
| E7 | Ostatnia nieliniowość: średnia(LGT, GLT) − średnia(pozostałe 4 permutacje); para LGT vs GLT (kolejność przy tym samym wyjściu tanh) |
| E8 | Udział grup BANK w dobroci i gradiencie |

### 7.8. Wrażliwość
- **(a)** Werdykty §7.3–7.4 z dwuwymiarowego bootstrapu (losowanie z powtórzeniami ziaren i, niezależnie, przykładów testowych; 10 000 replikacji; percentylowy; por. R3-04):
  - CI_K = kwantyle (α'/2, 1 − α'/2);
  - CI2_K = kwantyle (α', 1 − α');
  - inf(·): jednostronne p bootstrapowe = odsetek replikacji po złej stronie progu, następnie Holm jak w §7.3;
  - niezależny bootstrap po ziarnach i przykładach w układzie skrzyżowanym jest zachowawczy. Jest to akceptowalne dla analizy, która może tylko osłabić twierdzenie.

  Twierdzenia o „rozkładzie MNIST/F-MNIST” (a nie o zbiorze testowym) wolno formułować tylko wtedy, gdy werdykt (a) jest taki sam jak główny.
- **(b)** Test Wilcoxona zamiast t.
- **(c)** Wyniki biegów pierwotnych zastąpionych powtórkami przed ewaluacją oraz wyniki powtórek po ewaluacji, jeśli takie były (§5.6).

---

## 8. Wykonanie
- **Sprzęt:** GPU w Vast.ai (D1). Typ GPU i obraz Dockera ustalone przed mini-pilotem i stałe we wszystkich etapach.
  - Zmiana typu w trakcie etapu jest zabroniona.
  - Awaria instancji = nowa instancja tego samego typu i wznowienie przerwanych biegów od zera.
  - Zmiana typu między etapami (np. brak dostępności) wymaga przed startem kolejnego etapu powtórzenia testu 9.1.6 i pomiaru czasu z §9.3 na nowym sprzęcie, z wynikami w `ODSTEPSTWA.md` (por. R3-03).
- **Dane i wyniki:** przed zniszczeniem instancji wszystkie wyniki etapu (JSON-y, checkpointy, predykcje, logi) są kopiowane lokalnie, a ich hashe porównywane i commitowane. Każde uruchomienie i zniszczenie instancji oraz każde doładowanie konta wymaga potwierdzenia autora.
- **Kolejność:** dla każdego ziarna ramiona i zbiory w losowej kolejności (permutacja ze strumienia ziarna).
- **Atomowość:** przerwany bieg (np. przerwa na zajęcia) uruchamiany od zera z tym samym ziarnem.
- **Zapis per bieg:** JSON z metrykami walidacyjnymi, krzywymi, wersjami i hashem checkpointu. **Bez dokładności testowej.**
- **Monitor:** tylko liczba ukończonych biegów, liczba błędów i NaN.

---

## 9. Weryfikacja przed zamrożeniem

Kod (runner, analiza, symulacja mocy) jest pisany **po** akceptacji tego protokołu i na jego podstawie (por. G4).

### 9.1. Testy jednostkowe
1. Liczby parametrów = §3.
2. Każda warstwa liniowa (oprócz odczytu BP) ma po sobie nieliniowość.
3. Etykieta negatywna ≠ prawdziwa w 100% przypadków.
4. Uczenie warstwy 2 nie zmienia parametrów warstwy 1.
5. W trybie treningowym nie da się wczytać zbioru testowego.
6. Bitowa powtarzalność (ramię, ziarno), wykonana na docelowym sprzęcie.
7. Identyczne strumienie `split`/`order`/`neg` dla różnych ramion, także w warstwie 2 przy różnych epokach zatrzymania warstwy 1.
8. Odsetek obrazów z niezerowymi pikselami 0–9 per zbiór. Jeśli > 1%, wpis do §13.
9. Buforowanie (§5.1) daje bitowo ten sam wynik co przeliczanie warstwy 1.

### 9.2. Test skryptu analizy na symulacji całego potoku (por. R1-21)
Dane syntetyczne o znanych średnich: 16 ramion, 2 zbiory, N ziaren, korelacje jak w §7.5, z biegami NaN. Kontrole (każda 2 000 symulacji):

| | Prawdziwy stan | Wymaganie |
|---|---|---|
| (a) | Δ = 0 wobec jednego komparatora, 2S wobec pozostałych | odsetek werdyktów w klasie P ≤ α'/2 (z marginesem Monte Carlo) |
| (a′) | Δ = S wobec jednego komparatora, 3S wobec pozostałych | odsetek kategorii PRZEWAGA ≤ α'/2 |
| (b) | Δ = 0 wobec wszystkich | odsetek werdyktów w klasie G ≤ α'/2 |
| (c) | scenariusz (i) z §7.5 | moc klasy P zgodna z §7.5 |
| (d) | scenariusz (ii) z §7.5 | moc klasy R zgodna z §7.5 |
| (e) | dowolny | przypisanie kategorii §7.3–7.4 zgodne z implementacją referencyjną na 100 ręcznie skonstruowanych przypadkach, w tym RÓWNOWAŻNOŚĆ+, wszystkie kombinacje łączenia zbiorów i NaN |

### 9.3. Mini-pilot pomiarowy
**Cel:** pomiar wielkości potrzebnych do §5.2, §7.5 i §11. Bez wniosków o hipotezach.

**Ustawienia:**
- ramiona `P-R` i `K-LGT`;
- konfiguracja stała: lr 1e-3, wd 1e-4, c = 0.25;
- E_max = 60;
- ziarna 40001–40010;
- oba zbiory;
- docelowy sprzęt (typ GPU z Vast.ai wybrany wg D1, ten sam co w Etapach A–C);
- **tylko podział treningowo-walidacyjny, zbiór testowy nie jest używany.**

**Produkuje:**
1. σ_d na obu zbiorach (SD różnic `K-LGT` − `P-R` po ziarnach),
2. błędy walidacyjne `K-LGT` i `P-R` → SESOI (§7.5),
3. rozkład epok najlepszych → E_max,
4. czas na epokę i na bieg → budżet §11,
5. składową wariancji z próbkowania przykładów (na walidacji 10k) → raport w §7.5.

**Wyniki** zapisywane w `MINIPILOT.json` i raportowane. Nie wchodzą do analizy głównej.

**Jednokrotność** (por. R2-09):
- mini-pilot wykonywany jest **jeden raz**;
- surowy `MINIPILOT.json` jest commitowany **przed** wyliczeniem N, S, E_max i budżetu;
- runner mini-pilota i skrypt wyliczający N, S, E_max i budżet są commitowane **przed** uruchomieniem mini-pilota;
- przerwany bieg (awaria, przerwa na zajęcia) jest uruchamiany od zera z tym samym ziarnem (§8);
- powtórzenie całego mini-pilota dopuszczalne tylko po awarii technicznej, z wpisem w `ODSTEPSTWA.md`.

**Sprzęt i koszt:** sprzęt jest ustalony decyzją D1 (Vast.ai). Mini-pilot mierzy dodatkowo:
- koszt w USD na komplet ramion,
- liczbę biegów równoległych na karcie, przy której przepustowość jest największa (liczba ustalana przed mini-pilotem pomiarem przepustowości na danych losowych, a następnie stała).

### 9.4. Szablon raportu (results-blind)
`RAPORT_SZABLON.md`:
- wszystkie tabele z pustymi polami;
- dla H1, H2, H3 wszystkie kategorie z §7.3–7.4 i gotowe zdania z §7.4;
- sekcja eksploracyjna z nagłówkiem „wyniki eksploracyjne – bez wniosków konfirmacyjnych”.

### 9.5. Zasady języka raportu
- Bez słów „dowodzi”, „bezprecedensowy”, „idealny”, „rozwiązuje”, „deklasuje”.
- Każdy wniosek z zakresem: „MLP, MNIST/Fashion-MNIST, schemat FF z §5, na oficjalnym zbiorze testowym”.
- Wynik negatywny lub nierozstrzygnięty opisywany wprost, na pierwszej stronie.

---

## 10. Zamrożenie i znaczniki czasu

1. Po akceptacji protokołu i ukończeniu §9 (w tym uzupełnieniu §5.2, §7.5, §11 wartościami z mini-pilota):
   - commit z tagiem `ff-od-zera-prereg`,
   - `SHA256SUMS` (protokół, runner, analiza, symulacja mocy, szablon raportu, `MINIPILOT.json`).
2. **Zewnętrzny znacznik czasu** (por. G1): rejestracja na OSF albo tag w publicznym repozytorium, **przed Etapem A**. Jest to warunek uznania badania za prerejestrowane. Bez niego raport nazywa badanie „zaplanowanym z góry, bez zewnętrznej rejestracji”.
3. **Po Etapie B:** commit + hash `ETAP_B_WYBOR.json`, z tym samym mechanizmem znacznika czasu, przed Etapem C.
4. **Po Etapie C (trening):** commit hashy checkpointów przed ewaluacją testową (§5.6).
5. **`ODSTEPSTWA.md`:** każda zmiana po zamrożeniu z datą, treścią, powodem i informacją, czy nastąpiła przed czy po ewaluacji testowej.

---

## 11. Budżet

### 11.1. Limit finansowy (D1)
**Twardy limit: 38 USD łącznie** (saldo Vast.ai 13.99 USD + doładowanie najwyżej 100 PLN ≈ 24 USD po przewalutowaniu).

| Pozycja | Limit |
|---|---|
| Mini-pilot, pomiary, testy §9 | 5 USD |
| Etapy A + B | 12 USD |
| Etap C + E2 | 18 USD |
| Rezerwa (awarie, powtórki §5.6) | 3 USD |

### 11.2. Szacunek obciążenia (do przeliczenia po mini-pilocie)
Jednostka: komplet = 16 ramion FF × (1 ziarno, 1 zbiór).

| Etap | Komplety |
|---|---|
| A | 32 |
| B (z regułą brzegową, zakładaną domyślnie) | 18 + ≤ 24 |
| C | 2N (N ∈ {30, 40, 50, 60}) |
| E2 | 20 biegów BP |
| Mini-pilot | 40 biegów |

Odniesienie: na 3 rdzeniach i5-3450 komplet to ~75–110 min przy średnio 20–30 epokach. Całość to ~200–290 h CPU przy N = 40, w najgorszym przypadku ~330–480 h CPU.

**Po mini-pilocie** (por. R3-03): **cała** tabela (wszystkie etapy, nie tylko C) jest przeliczana z:
- rozkładu epok,
- E_max,
- czasu na epokę,
- liczby biegów równoległych,
- ceny za godzinę wybranej instancji.

Przeliczenie podaje koszt w USD dla każdego N oraz scenariusz najgorszy (N = 60, podniesienie E_max o 50%) i jest wpisywane do protokołu przed zamrożeniem.

### 11.3. Zasady
- **Przed zamrożeniem:** jeśli przewidywany koszt przy N z reguły §7.5 przekracza 33 USD (limit bez rezerwy), stosowane są redukcje w tej kolejności:
  1. pominięcie E2;
  2. Etap A: 8 punktów Sobola (pierwsze 8 z sekwencji);
  3. tańszy typ GPU spełniający wymagania D1, jeśli pomiar wykaże niższy koszt na komplet.
- Redukcja N poniżej wartości z reguły §7.5 jest niedopuszczalna. Jeśli po redukcjach koszt nadal przekracza limit, badanie nie rozpoczyna się, a autor decyduje o budżecie przed zamrożeniem.
- **Po zamrożeniu:** przekroczenie limitu w trakcie (np. po podniesieniu E_max w Etapie B) powoduje **wstrzymanie** badania, a nie jego skrócenie. Decyzja autora o dodatkowym budżecie trafia do `ODSTEPSTWA.md` przed wznowieniem. Zmniejszanie N, liczby ramion lub E_max jest niedopuszczalne.
- Instancja jest niszczona po zakończeniu każdego etapu (§8).

---

## 12. Identyfikowalność

Numery sekcji w tabelach 12.1–12.3 odnoszą się do wersji wskazanej w nagłówku tabeli. W v4 budżet przeniesiono do §11.1–11.3 (D1).

### 12.1. Audyt (S*, L*, K*, G*) → v2

| ID | Naprawa |
|---|---|
| S1, S5 | §5.2, §10.5 |
| S2, K16 | §7.3, §7.5, §9.2 |
| S3 | §7.2 |
| S4, G3 | §9.4 |
| S6 | §3 |
| S7, L5, K1, K2 | §5.3–5.4 |
| S8 | §7.2 |
| S9, K14 | §6.2 |
| S10, K12, G5 | §9.5 |
| L1 | §3.1, §9.1.2 |
| L2 | §3.3 (P vs K; monokultury K) |
| L3 | audyt |
| L4, K15 | §6.2 |
| L6 | §1 |
| L7, K18 | §4.3 |
| K3 | §5.6, §7.5, §8 |
| K4, K5 | §2, §7.4, §9.4 |
| K6 | §7.3 |
| K7 | §7.7 |
| K8 | §3.3, E2 |
| K9 | §4.2, §5.3–5.6 |
| K10 | §3.3, §2 |
| K11 | §7.2 |
| K13 | §4.2–4.3, §7.8 |
| K17 | §6.2 |
| G1 | §10.2 |
| G2 | §1 |
| G4 | §9.1–9.2 |
| G6 | §14 |

### 12.2. Recenzja R1 → v2

| ID | Naprawa |
|---|---|
| R1-01 | §2, §7.3, §13.6 |
| R1-02 | §1, §2, §3.3, E7, §13 |
| R1-03 | §5.3, §5.4, E5, §13 |
| R1-04 | §5.2, §6.2 |
| R1-05 | §4.2, §5.1, §11 |
| R1-06 | §7.5, §9.3, §11.2 |
| R1-07 | §7.1, §7.8, §9.3 |
| R1-08 | §5.5–5.6, §8, §10.4 |
| R1-09 | §7.3–7.4 |
| R1-10 | §5.1, §7.6 |
| R1-11 | §5.7 |
| R1-12 | §1, §6.2, E8, §13 |
| R1-13 | §7.5 |
| R1-14 | §5.1, E6, §13 |
| R1-15 | §3.3 |
| R1-16 | §4.3, §9.1.7 |
| R1-17 | §5.3, E2 |
| R1-18 | §6.2 |
| R1-19 | §5.7, §8, §9.1.6 |
| R1-20 | §9.1.8 |
| R1-21 | §9.2 |

### 12.3. Recenzja R2 → v3

| ID | Naprawa |
|---|---|
| R2-01 | §7.5 (cele mocy (i)/(ii), PRZEWAGA drugorzędna, N ∈ {30…60}), §11.1 (koszt per N) |
| R2-02 | §7.3 (RÓWNOWAŻNOŚĆ+), §7.4 (klasy, łączenie, dopiski), §9.2(e) |
| R2-03 | §7.4 (komórka R×P) |
| R2-04 | §5.6 p. 4, §7.8(c) |
| R2-05 | §7.5 (S per topologia) |
| R2-06 | §7.3 (Holm na p, α'/2) |
| R2-07 | §5.4 (reguła brzegowa), §11.1 |
| R2-08 | §9.2 (a), (a′), (e) |
| R2-09 | §9.3 (jednokrotność) |
| R2-10 | §4.2, §11.1, §11.2, §12.2 |


### 12.4. Recenzja R3 i decyzja D1 → v4

| ID | Naprawa |
|---|---|
| R3-01 | §7.5 (σ_d × 1.29 w regule N; moc przy × 1.47 raportowana) |
| R3-02 | §7.5 (m = 5, S = min, σ_d jak wyżej) |
| R3-03 | §8 (zmiana sprzętu między etapami), §11.2 (przeliczenie całej tabeli) |
| R3-04 | §7.8(a) |
| D1 | §5.7, §8, §9.3, §11 |

---

## 13. Ograniczenia znane przed badaniem

1. Tylko MLP i małe zbiory obrazów w skali szarości.
2. Jeden schemat FF: etykieta w pikselach, losowe negatywy, uczenie zachłanne, suma dobroci z obu warstw. Inne warianty FF mogą dawać inne wyniki.
3. Ramiona K używają lokalnej backpropagacji wewnątrz bloku, czyli uczenia lokalnego blokowego, a nie FF warstwa-po-warstwie (Nøkland i Eidnes, 2019; Belilovsky i in., 2019).
4. **Strojenie:**
   - wyniki są warunkowe względem przestrzeni i budżetu strojenia (16 punktów, wspólne c dla obu warstw FF);
   - dla ramion z wyjściem ReLU/LeakyReLU c jest częściowo wymienne ze skalą wag, więc efektywny wymiar ich strojenia jest mniejszy;
   - dla ramion z wyjściem tanh część zakresu c (blisko 1) prowadzi do nasycenia;
   - budżet jest jednak równy, a krajobraz raportowany (E5).
5. **Ograniczona dobroć tanh:** ramiona z wyjściem tanh (`P-T`, `K-T`, `K-LGT`, `K-GLT`) mają G ≤ 250, pozostałe nie. Wyniki H1/H2 mogą częściowo odzwierciedlać typ ostatniej nieliniowości (E7).
6. **Komparatory** w H1–H3 są oceniane wszystkie, bez selekcji. Werdykt PRZEWAGA wymaga przewagi nad każdym z nich (§7.3).
7. **Zbiór testowy jest stały.** Wnioski główne są warunkowe względem niego (§7.1). Wrażliwość w §7.8.
8. **Inicjalizacja** domyślna PyTorch dla wszystkich ramion (§5.7), nieoptymalna dla części nieliniowości.
9. **BANK:** jednostki tanh mają ograniczony wkład do Σa² (E8).

---

## 14. Literatura

- Belilovsky, E., Eickenberg, M., & Oyallon, E. (2019). Greedy layerwise learning can scale to ImageNet. *ICML 2019*, PMLR 97, 583–593.
- Berger, R. L., & Hsu, J. C. (1996). Bioequivalence trials, intersection-union tests and equivalence confidence sets. *Statistical Science*, 11(4), 283–319.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281–305.
- Bouthillier, X., i in. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems 3 (MLSys 2021)*.
- Dodge, J., Gururangan, S., Card, D., Schwartz, R., & Smith, N. A. (2019). Show your work: Improved reporting of experimental results. *EMNLP-IJCNLP 2019*, 2185–2194.
- Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv:1606.08415*.
- Hinton, G. (2022). The Forward-Forward algorithm: Some preliminary investigations. *arXiv:2212.13345*.
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Melis, G., Dyer, C., & Blunsom, P. (2018). On the state of the art of evaluation in neural language models. *ICLR 2018*.
- Nøkland, A., & Eidnes, L. H. (2019). Training neural networks with local error signals. *ICML 2019*, PMLR 97, 4839–4850.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. *arXiv:1708.07747*.
