# Raport: heterogeniczność i kolejność nieliniowości w sieciach Forward-Forward

> Szablon napisany przed wynikami (PROTOKOL_v5 §9.4). Pola `{…}` wypełnia skrypt `ffexp/final.py` albo autor, przepisując liczby bez zmian. Zdania interpretacji pochodzą wyłącznie z tabel poniżej. Wszystko spoza tych tabel trafia do części „Wyniki eksploracyjne”.
>
> Słowa zakazane (§9.5): „dowodzi”, „bezprecedensowy”, „idealny”, „rozwiązuje”, „deklasuje”.

**Status badania:** {prerejestrowane z zewnętrznym znacznikiem czasu: link | zaplanowane z góry, bez zewnętrznej rejestracji}; {ukończone | nieukończone (§5.6 pkt 6)}
**Zakres wszystkich wniosków:** MLP, MNIST/Fashion-MNIST, schemat FF z §5 protokołu, na oficjalnym zbiorze testowym.

## Najważniejsze wyniki (pierwsza strona)

- **H1:** {kategoria łączna} – {zdanie z macierzy}
- **H2:** {kategoria łączna} – {zdanie z macierzy}
- **H3:** {kategoria łączna} – {zdanie H3}
- Wyniki nierozstrzygnięte lub niespójne: {lista, opisana wprost}
- Rozbieżność MNIST-10k vs QMNIST-50k (§7.8d): {brak | lista hipotez z różnym werdyktem}
- Odstępstwa od protokołu: {brak | odesłanie do ODSTEPSTWA.md, z liczbą pozycji}

## 1. Wykonanie

| Element | Wartość |
|---|---|
| N (ziarna Etapu C) | {N} |
| S: H1/H2 – MNIST, F-MNIST | {S} p.p., {S} p.p. |
| S: H3 – MNIST, F-MNIST | {S} p.p., {S} p.p. |
| E_max (Etapy A, B, C) | {…}, {…}, {…} |
| Reguła brzegowa | {osie, liczba punktów} |
| GPU, obraz, wersje | {…} |
| Koszt łączny | {USD} z 38 USD |
| Biegi z NaN (per ramię) | {tabela} |
| Odsetek biegów z aktywnym limitem epok | {tabela} |

## 2. Dokładność testowa (średnia ± SD po N ziarnach)

| Ramię | MNIST | F-MNIST |
|---|---|---|
| {18 wierszy} | {…} | {…} |

## 3. Hipotezy konfirmacyjne

Dla każdej hipotezy i każdego zbioru: kategoria (§7.3), średnie Δ z (1 − α') PU względem każdego komparatora, informacja o równoważności, wynik testu Wilcoxona.

### H1 (`K-LGT` vs `K-R`, `K-L`, `K-G`, `K-T`, `K-LLT`, `K-GGT`)
| Komparator | Δ MNIST [PU] | Δ F-MNIST [PU] |
|---|---|---|
| {6 wierszy} | | |

Kategoria MNIST: {…}; F-MNIST: {…}; **łącznie: {…}**

### H2 (`K-LGT` vs 5 pozostałych permutacji)
{jak wyżej}

### H3 (`P-BANK` vs `P-R`, `P-L`, `P-G`, `P-T`)
{jak wyżej}

### Interpretacja (wyłącznie zdania z poniższych tabel)

**Macierz H1 × H2** (P = PRZEWAGA lub PRZEWAGA<S; R = RÓWNOWAŻNOŚĆ lub RÓWNOWAŻNOŚĆ+; G = GORSZOŚĆ lub GORSZOŚĆ<S; N = NIEROZSTRZYGNIĘTY lub NIESPÓJNY-*). „Bloki referencyjne” to cała grupa komparatorów H1.

| H1 \ H2 | P | R | G | N |
|---|---|---|---|---|
| **P** | LGT lepszy od każdego bloku referencyjnego (także `K-T`, `K-LLT`, `K-GGT`) i od każdej innej kolejności. | LGT lepszy od każdego bloku referencyjnego; równoważny każdej innej kolejności (±S). | LGT lepszy od każdego bloku referencyjnego; co najmniej jedna inna kolejność lepsza od LGT. | LGT lepszy od każdego bloku referencyjnego; porównanie z innymi kolejnościami nierozstrzygnięte. |
| **R** | LGT równoważny blokom referencyjnym (±S) i lepszy od każdej innej kolejności; porównanie innych kolejności z blokami referencyjnymi jest eksploracyjne (E3, E7). | LGT równoważny każdemu blokowi referencyjnemu i każdej innej kolejności (±S). | LGT równoważny blokom referencyjnym; co najmniej jedna inna kolejność lepsza od LGT. | LGT równoważny blokom referencyjnym; porównanie z innymi kolejnościami nierozstrzygnięte. |
| **G** | LGT gorszy od co najmniej jednego bloku referencyjnego i lepszy od każdej innej kolejności. | LGT gorszy od co najmniej jednego bloku referencyjnego; pozostałe kolejności równoważne LGT (±S). | LGT gorszy od co najmniej jednego bloku referencyjnego i od co najmniej jednej innej kolejności. | LGT gorszy od co najmniej jednego bloku referencyjnego; porównanie z innymi kolejnościami nierozstrzygnięte. |
| **N** | Porównanie z blokami referencyjnymi nierozstrzygnięte; LGT lepszy od każdej innej kolejności. | Porównanie z blokami referencyjnymi nierozstrzygnięte; inne kolejności równoważne LGT. | Porównanie z blokami referencyjnymi nierozstrzygnięte; co najmniej jedna inna kolejność lepsza od LGT. | Badanie nie rozstrzyga H1 ani H2. |

Przy H1 = P dopuszczalne jest zdanie: „przewaga LGT nie sprowadza się do obecności tanh na wyjściu bloku referencyjnego (komparatory `K-T`, `K-LLT`, `K-GGT`)”.

**H3:**

| Klasa | Zdanie |
|---|---|
| P | Bank (Σa² po wszystkich jednostkach) lepszy od każdej pojedynczej nieliniowości w topologii płaskiej. |
| R | Bank równoważny każdej pojedynczej nieliniowości (±S). |
| G | Co najmniej jedna pojedyncza nieliniowość lepsza od banku. |
| N | Nierozstrzygnięty. |

**Obowiązkowe dopiski:**
- Przy kategoriach „<S”: „różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane”.
- Przy RÓWNOWAŻNOŚCI+: „wykryto różnicę istotną statystycznie, ale praktycznie pomijalną (< S)”, z kierunkiem na **każdym** zbiorze: MNIST {+/−}, F-MNIST {+/−}.
- Przy NIESPÓJNY-ZGODNY-KIERUNEK: „na zbiorze {X}: {kategoria zbiorowa}; na zbiorze {Y}: różnica istotna statystycznie w tym samym kierunku, praktycznie pomijalna (< S, równoważność wykazana)”.
- Przy NIESPÓJNY-KIERUNEK: „kierunek różnicy przeciwny na obu zbiorach: {X}: {…}, {Y}: {…}”.
- Przy klasie G w H1 lub H2: lista komparatorów istotnie lepszych od LGT, wraz z ich nieliniowością wyjściową: {…}.
- Przy NIEROZSTRZYGNIĘTY (niestabilność): „werdykt testu t ({…}) i testu Wilcoxona ({…}) różnią się w obecności biegów z NaN ({liczba})”.

## 4. Analizy wrażliwości (§7.8)

| Hipoteza | Główny | (a) bootstrap 2D | (b) Wilcoxon | (c) biegi pierwotne | (d) QMNIST-50k (tylko MNIST) |
|---|---|---|---|---|---|
| H1 | {…} | {…} | {…} | {nie dotyczy / …} | {…} |
| H2 | | | | | |
| H3 | | | | | |

Twierdzenia o rozkładzie danych (a nie tylko o zbiorze testowym): {dopuszczalne dla …, gdzie (a) zgodne z głównym | niedopuszczalne}.

---

## 5. Wyniki eksploracyjne – bez wniosków konfirmacyjnych

> W tej części wolno przypisywać mechanizmy (np. „różnorodność”, „kolejność”, „ograniczona dobroć tanh”), zawsze jako hipotezy do dalszego sprawdzenia.

- **E1** – głębokość (K vs P): {tabela}
- **E2** – luka do BP (N = 10; BP strojone w 2D, FF w 3D; na F-MNIST FF traci piksele 0–9, §13.14): {tabela}
- **E3** – kontrasty pojedyncze, `K-BANK`, inne kolejności vs bloki referencyjne: {tabela}
- **E4** – metryki drugorzędne: {…}
- **E5** – krajobraz strojenia (nieosiągalny zakres c dla wyjść tanh zaznaczony): {rysunek}
- **E6** – inferencja tylko z G₂: {tabela}
- **E7** – wyjście tanh: {tabela}
- **E8** – udział grup BANK w dobroci i gradiencie: {…}
- **E9** – przegląd szerokości. Przewidywanie autora zapisane przed badaniem: przewaga ramion heterogenicznych rośnie, gdy w maleje. Wynik: {opis bez oceny „potwierdzone”}. Rysunek 2.
- **E10** – spirala 2D: najmniejszy budżet z medianą dokładności ≥ 95%: {tabela}. Rysunek 1 (mapy decyzyjne, w = 16, ziarno o dolnej medianie dokładności walidacyjnej spośród 10; opis zawiera ramię, budżet, liczbę parametrów, ziarno i jego dokładność testową).

## 6. Ograniczenia

Pełna lista: PROTOKOL_v5 §13 (1–14), a także odstępstwa z `ODSTEPSTWA.md`.
