# R6 – recenzja raportu końcowego (`RAPORT.md`, stan z commitu 7edf5cf)

Recenzent niezależny, metodologiczny. Data: 2026-10-24.

## Zakres i metoda

Przeliczenia niezależne wykonałem z plików predykcji `wyniki/test/{C,E2,E9}/*.pt` (4720 plików) i rekordów `wyniki/{C,E2,E9}/runs/*.json`. Napisałem do tego własny kod (scratchpad `r6/an.py`, `r6/e9.py`, `r6/e2.py`), który nie importuje `ffexp`. Testy t, procedurę Holma i TOST zaimplementowałem od nowa według §7.2–7.3. Nie uruchamiałem `final.py` ani `evaluate_test` i nic nie zapisałem w `wyniki/`. Katalog `zaslepione/` (walidacja Etapu C i E9) otworzyłem; po kroku 2 §5.6 jest to dozwolone.

**Co się zgadza (sprawdzone niezależnie):**
- Część 2: wszystkie 18 × 2 średnich i SD (N = 120), NaN = 0 we wszystkich ramionach.
- Część 3: wszystkie Δ i PU (1 − α′) dla H1–H3. Różnice wynikają tylko z zaokrągleń w 3. miejscu, np. `P-G` MNIST −0.9105.
- Kategorie per zbiór, kategorie łączne i flaga `eq`. Zdanie z macierzy G × R i zdanie H3 = G są wybrane poprawnie. Łączenie §7.4 jest poprawne, w tym dla H3: GORSZOŚĆ + GORSZOŚĆ<S → GORSZOŚĆ<S.
- QMNIST-50k (d): H1 GORSZOŚĆ<S, H2 RÓWNOWAŻNOŚĆ+, H3 GORSZOŚĆ.
- Kategorie (a) i (b) w części 4 zgadzają się z `WYNIKI_KONCOWE.json`. Bootstrapu nie liczyłem ponownie.
- Liczby ze źródeł:
  - E1 (z wyjątkiem R6-13), E2, E3, E6 i E7 – zgodne;
  - tabela E9 – zgodna z `E9_width_curves`;
  - tabela E10 – zgodna z `e10_wyniki.json`;
  - zliczenia „epoki 0” warstwy 2 – zgodne;
  - 4 pozycje w `ODSTEPSTWA.md`;
  - koszt 13.99 − 3.83 = 10.16 USD.
- Słowa zakazane z §9.5 („dowodzi”, „bezprecedensowy”, „idealny”, „rozwiązuje”, „deklasuje”) nie występują. Nie ma też słów „ratuje” ani „potwierdza”.

Poniżej uwagi. Wagi: **krytyczna** blokuje pokazanie raportu, **istotna** trzeba poprawić przed wersją końcową, **drobna** to poprawka redakcyjna lub uzupełnienie.

---

## Uwagi

### R6-01 (krytyczna) – `K-R` nie jest „istotnie lepszy od LGT” według kryterium protokołu

**Lokalizacja:**
- pierwsza strona, punkt H1 („`K-R` (wyjście ReLU; tylko MNIST)”);
- część 3, obowiązkowy dopisek „Klasa G w H1 – komparatory, których PU (1 − α′) dla Δ leży całkowicie poniżej zera…”.

**Dowód:**
- §7.3 definiuje klasę G przez inf(0): jednostronne p_K dla H₀: Δ_K ≥ 0, procedura Holma na m = 6 wartościach p, poziom α′/2 = 0.00833.
- Kod (`analysis.py:69–70`, `categorize_from_p`) sprawdza tylko warunek `min p ≤ (α′/2)/m`. Nie zwraca listy odrzuconych komparatorów, więc listy w raporcie nie wyliczył `final.py`. Autor wyznaczył ją ręcznie, innym kryterium.
- Kryterium z raportu, „PU (1 − α′) całkowicie poniżej zera”, to nieskorygowany test jednostronny na poziomie α′/2 dla każdego komparatora osobno. Nie jest to inf(0).
- Moje przeliczenie (test t, n = 120) daje następujące kroki Holma.

| Zbiór | Krok 1 (próg 0.00833/6 = 0.00139) | Krok 2 (próg 0.00833/5 = 0.00167) |
|---|---|---|
| MNIST | `K-GGT`: p = 6.1·10⁻⁹ → odrzucona | `K-R`: p = 3.9·10⁻³ > 0.00167 → **stop** |
| F-MNIST | `K-GGT`: p = 4.0·10⁻⁴ → odrzucona | `K-R`: p = 0.16 → stop |

- Zbiór komparatorów istotnie lepszych od LGT to więc **tylko `K-GGT` (wyjście tanh), na obu zbiorach**. Zdanie o `K-R` przekracza kontrolę błędu z §7.2.

**Skutek interpretacyjny:**
- Jedyny istotnie lepszy komparator ma wyjście tanh, tak samo jak LGT.
- Zastrzeżenie §13.5 (przegrana z komparatorem o nieograniczonym wyjściu) nie tłumaczy więc wyniku H1. Wolno to napisać wyłącznie w części eksploracyjnej.
- Na QMNIST-50k Holm odrzuca `K-GGT`, `K-R` i `K-T`. Można to podać w części 4 jako wynik wrażliwości, ale nie wolno przenosić tych komparatorów do listy konfirmacyjnej.

**Poprawka:**
- Na pierwszej stronie i w dopisku wpisać: „Komparatory istotnie lepsze od LGT (Holm, §7.3): `K-GGT` (wyjście tanh) – MNIST i F-MNIST”.
- Zastąpić kryterium „PU całkowicie poniżej zera” kryterium Holma.
- Najlepiej dodać do tabel H1–H3 kolumnę z wynikiem Holma (odrzucona / nie) albo z p skorygowanym metodą Holma. Sam PU (1 − α′) nie pokazuje, które porównania są istotne w sensie inf(0).

### R6-02 (istotna) – kierunek „−” przy RÓWNOWAŻNOŚCI+ w H2 ukrywa kierunki mieszane

**Lokalizacja:** pierwsza strona, punkt H2; część 3, dopisek H2.

**Dowód:**
- W kodzie `direction = +1` tylko przy sup(0), który jest IUT po wszystkich komparatorach. `direction = −1` zachodzi przy inf(0), czyli teście sumy z Holmem (`analysis.py:74`). Znak „−” oznacza więc tylko, że co najmniej jeden komparator jest istotnie lepszy od LGT. Istotnych różnic w drugą stronę nie wyklucza.
- Holm, inf(0):
  - MNIST: odrzucone `K-TLG`, `K-LTG`, `K-TGL`;
  - F-MNIST: odrzucony tylko `K-TGL` (p = 9.8·10⁻⁴ < 0.00167).
- Na F-MNIST LGT jest jednocześnie istotnie lepszy od `K-LTG` (+0.815 [+0.596, +1.034]) i od `K-GTL` (+0.283 [+0.096, +0.469]). Holm dla H₀: Δ ≤ 0 odrzuca obie hipotezy.
- Interpretacja „(co najmniej jedna inna kolejność nieznacznie lepsza od LGT)” jest formalnie poprawna. Pomija jednak, że na F-MNIST różnice idą w obu kierunkach.
- Słowo „nieznacznie” jest dodatkiem redakcyjnym. Protokół mówi: „praktycznie pomijalną (< S)”.

**Poprawka:**
- Przy kierunku podać komparatory, które go wyznaczają: „MNIST −: `K-LTG`, `K-TLG`, `K-TGL`; F-MNIST −: `K-TGL`”.
- Dodać jedno zdanie opisowe: „na F-MNIST jednocześnie LGT istotnie lepszy od `K-LTG` i `K-GTL`; wszystkie różnice w paśmie ±S”.
- Zamienić „nieznacznie” na „(< S)”.

### R6-03 (istotna) – „Odsetek biegów z aktywnym limitem epok” policzony według innej definicji niż §5.2

**Lokalizacja:** część 1, ostatni wiersz; część 2, kolumna „limit E_max”.

**Dowód:**
- §5.2 definiuje aktywny limit jako „najlepszą epokę ≥ E_max − 2 w którejkolwiek warstwie”. Tak samo liczy `plans.py:183–186`, który dał `ETAP_A_WYBOR.json` i `ETAP_B_WYBOR.json`.
- Według tej definicji w Etapie C (E_max = 77, `best_epoch` indeksowane od 0):
  - `P-T` MNIST: **11/120** (9.2%). Przy indeksowaniu od 1: 14/120;
  - `K-L` MNIST: **0/120**.
- Liczby z raportu, 21/120 i 1/120, to biegi, w których `epochs_run` osiągnęło E_max. Mierzą wyczerpanie limitu, a nie §5.2.
- Ani reguła limitu, ani inne wnioski nie zmieniają się: obie wartości są poniżej 25%. Raport podaje jednak liczbę pod nazwą zdefiniowaną w protokole, a liczoną inaczej.

**Poprawka:** podać obie miary z definicjami, np. „`P-T` MNIST: najlepsza epoka ≥ E_max − 2: 11/120; trening do E_max: 21/120”. To samo w kolumnie części 2.

### R6-04 (istotna) – E9: „najlepsze ramię referencyjne” można wyznaczyć według protokołu, a wynik jest nierozstrzygający

**Lokalizacja:** część 5, E9.

**Dowód:**
- Raport wybiera najlepszy blok referencyjny według dokładności **testowej**. Twierdzi, że wybór według definicji §7.7 (średnia walidacyjna z `zaslepione/`) „pozostaje do uzupełnienia”.
- Dane są dostępne: `wyniki/E9/zaslepione/*.json` i `wyniki/C/zaslepione/*.json`, pole `val_acc`, ziarna 10001–10010.
- Według walidacji najlepszy jest **`K-GGT` przy każdym w, na obu zbiorach**. Przykład: MNIST w = 32, walidacja `K-GGT` 94.824, `K-LLT` 94.799, `K-R` 94.387.
- Raport przy MNIST w = 32 wybrał `K-LLT` (test 94.89). Liczba „−0.75” nie odpowiada więc definicji z protokołu. Według definicji wynosi **−0.68**.
- Różnica „w = 32 minus w = 250” z moim przedziałem 95% (bootstrap po ziarnach, różnice sparowane między szerokościami; §7.7 nie ustala metody, to przedział recenzenta):

| Kontrast | MNIST | F-MNIST |
|---|---|---|
| `K-LGT` − najlepszy (`K-GGT`) | −0.45 [−1.15, +0.35] | −0.04 [−1.11, +0.90] |
| `P-BANK` − `P-R` | −1.31 [−1.71, −0.89] | −6.98 [−9.39, −4.58] |

- Dla `K-LGT` przedziały obejmują 0 i sięgają wartości dodatnich, aż do +0.9 p.p. na F-MNIST. Zdanie „nie jest większa niż przy w = 250” opisuje tylko estymatę punktową. Przy N = 10 dane **nie rozstrzygają** przewidywania dla `K-LGT`.
- Dla `P-BANK` wynik jest wyraźnie przeciwny przewidywaniu.
- Zdanie „Kierunek wniosku nie zależy od wyboru komparatora” jest kruche. Dla `K-R` na F-MNIST różnica 32 − 250 wynosi −0.00 [−0.81, +0.69].
- §7.7 wymaga też „średnich z 95% przedziałami ufności” i różnic „względem **każdego** ramienia E9”. Raport podaje same średnie i odsyła do JSON. Różnic względem każdego ramienia nie ma nigdzie, także w `final.py`. Moje wartości przy w = 32 / 250 (95% CI t):

| Kontrast | MNIST w = 32 | MNIST w = 250 | F-MNIST w = 32 | F-MNIST w = 250 |
|---|---|---|---|---|
| `K-LGT` − `K-R` | −0.35 [−1.00, +0.31] | −0.05 [−0.29, +0.19] | −0.11 [−0.73, +0.51] | −0.10 [−0.87, +0.67] |
| `K-LGT` − `K-LLT` | −0.74 [−1.43, −0.06] | +0.84 [+0.54, +1.14] | +0.46 [−0.26, +1.19] | +1.48 [+0.60, +2.36] |
| `K-LGT` − `K-GGT` | −0.68 [−1.54, +0.18] | −0.23 [−0.47, −0.00] | −0.34 [−1.27, +0.59] | −0.30 [−1.01, +0.41] |

**Poprawka:**
- Przeliczyć E9 według definicji §7.7 (`K-GGT`), z przedziałem bootstrap i z jawnie opisaną metodą.
- Dodać tabelę różnic względem każdego ramienia E9 przy każdym w, z 95% CI.
- Sformułować wynik: „dla `K-LGT` dane nierozstrzygające (CI obejmuje 0); dla `P-BANK` różnica przy w = 32 bardziej ujemna niż przy w = 250, przeciwnie do przewidywania”.
- Dopisać, że kolumna w = 250 to podzbiór 10 ziaren (10001–10010) z Etapu C, stąd np. `K-LGT` MNIST 97.33 zamiast 97.36 z części 2.

### R6-05 (istotna) – E2: mniejsza luka dla K-LGT wynika częściowo ze słabszego BP-K-LGT

**Lokalizacja:** część 5, E2, zdanie pod tabelą.

**Dowód (przeliczenie niezależne, N = 10):**
- `BP-K-LGT` − `BP-P-R`: MNIST −0.30 [−0.49, −0.11], F-MNIST −1.09 [−1.45, −0.72]. Punkt odniesienia BP w topologii K jest więc istotnie słabszy.
- Różnica luk (luka K − luka P): MNIST −0.34 [−0.66, −0.02], F-MNIST −1.87 [−2.66, −1.08].
- Zdanie „95% CI obejmuje 0” przy N = 10 i szerokości przedziału ~1.7 p.p. na F-MNIST nie oznacza braku luki.

**Poprawka:**
- Dodać: „`BP-K-LGT` jest słabszy od `BP-P-R` (MNIST −0.30, F-MNIST −1.09 p.p.); mniejsza luka wynika częściowo ze słabszego punktu odniesienia BP. Przy N = 10 CI obejmujące 0 nie świadczy o braku luki”.
- Podać różnicę luk z CI.
- Przypomnieć §13.14: FF ma 10 wejść więcej niż BP.

### R6-06 (istotna) – E10: błędne „około 10 razy”, mylące nagłówki, pominięta niestabilność

**Lokalizacja:** część 5, E10.

**Dowód (`wyniki/E10/e10_wyniki.json`):**
- „Około 10 razy mniejszym budżecie niż `K-R`”:
  - `K-LGT`: 120 vs 1275 parametrów, ok. 10.6×;
  - `K-GGT`: **315** vs 1275, ok. **4×**.
  - Przy siatce czterech budżetów współczynnik jest i tak artefaktem rozdzielczości siatki.
- Nagłówki kolumn „u=8, 16, 32, 64” są błędne. To szerokości P, czyli w (§3.4). Topologia K ma u = 4, 7, 15, 29, np. 120 parametrów przy u = 4.
- `P-R` jest niemonotoniczny: mediana 96.8% przy w = 16, potem 71.8% i 75.5%. „Minimalny budżet u = 16” bez tej informacji wprowadza w błąd.
- Wynik zależy od mediany. Według średniej `K-LGT` nie osiąga 95% przy w = 8 (91.35%) ani przy w = 16 (93.85%; jedno ziarno ma 61%). `K-GGT` ma przy tych budżetach wyższą średnią: 94.5% i 98.1%. Zdanie „`K-LGT`, `K-GGT` osiągają ≥ 95%…” jest prawdziwe tylko dla mediany, a ta nie pokazuje niestabilności.
- Określenie „gładkie podwarstwy” dla `K-LGT` jest nieścisłe, bo L (LeakyReLU) nie jest gładka. Jest to też przypisanie mechanizmu w zdaniu opisowym.

**Poprawka:**
- Zmienić nagłówki na „w = 8 … 64 (P); K: u = 4, 7, 15, 29”.
- Usunąć „około 10 razy” albo napisać „`K-LGT` ok. 10×, `K-GGT` ok. 4× mniej parametrów niż `K-R` (na siatce 4 budżetów)”.
- Oznaczyć niemonotoniczność `P-R`.
- Dodać średnie lub minimum z 10 ziaren albo uwagę o rozrzucie.
- Mechanizm przenieść do zdania „Hipoteza…”.

### R6-07 (istotna) – brak elementów obowiązkowych: E5, Rysunek 2, opis Rysunku 1; E4 i E8 niewykonane

**Lokalizacja:** część 5 (E4, E5, E8, E9, E10); część 6.

**Dowód:**
- §7.7: „E5 | Krajobraz strojenia z Etapu A (**obowiązkowo**), z zaznaczonym nieosiągalnym zakresem c dla wyjść tanh”. §13.4: „krajobraz strojenia jest raportowany (E5)”. Raport: „do uzupełnienia”.
- §7.9 wymaga Rysunku 2 (krzywe E9). Raport: „do wykonania”.
- §7.9 i szablon wymagają, żeby opis Rysunku 1 zawierał ramię, budżet, liczbę parametrów, wybrane ziarno, jego dokładność testową oraz informację, że to ziarno medianowe spośród 10. Raport podaje tylko ścieżkę pliku.
  - Dane są w `e10_wyniki.json`, pole `figure_seed`. Przy w = 16: `K-LGT` ziarno 50010, 99.0%; `K-GGT` 50001, 100.0%; `K-R` 50010, 89.0%; `P-R` 50004, 97.5%; `P-BANK` 50003, 80.0%; `P-T` 50009, 61.5%; `BP-P-R` 50005, 100.0%.
- Plik `rysunekS_mapy_wszystkie.png` (materiał uzupełniający z §7.7) istnieje, ale raport go nie wymienia.
- E4 (metryki drugorzędne z §6.2: jednostki nieaktywne, participation ratio) i E8 nie zostały wykonane. To nie jest naruszenie zamrożenia, bo `final.py` ich nie liczy. Raport bez nich nie jest jednak kompletny według §7.7.

**Poprawka:**
- Wykonać E5 i Rysunek 2 jako analizy eksploracyjne z nowym, zacommitowanym skryptem. W raporcie zaznaczyć, że powstały poza zamrożonym `final.py`.
- Uzupełnić opis Rysunku 1.
- E4 i E8 wykonać albo wpisać do `ODSTEPSTWA.md` jako niewykonane, z powodem.
- Do czasu uzupełnienia oznaczyć raport na pierwszej stronie jako „wersja robocza – brak E4, E5, E8, Rysunku 2”.

### R6-08 (istotna) – pierwsza strona nie mówi wprost, że przewidywania autora się nie sprawdziły

**Lokalizacja:** „Najważniejsze wyniki”.

**Dowód:**
- §2: przewidywanie autora to „przewaga” dla H1, H2 i H3, a „wynik przeciwny przewidywaniu jest raportowany na równi ze zgodnym”.
- §9.5: „Wyniki negatywne i nierozstrzygnięte opisane wprost na pierwszej stronie”.
- Wynik: H1 i H3 są w klasie G, czyli **przeciwnie** do przewidywania. H2 wyszło R zamiast P. Na pierwszej stronie nie ma zdania, które by to zestawiało.
- §16 zapowiadał też, że H3 będzie najpewniej NIEROZSTRZYGNIĘTY (moc 0.25). Wyszedł werdykt rozstrzygający. Warto podać, że obserwowane SD różnic, np. 0.35 dla `P-BANK` − `P-R` na MNIST, były znacznie mniejsze niż założone σ_d = 1.84.

**Poprawka:** dodać punkt: „Przewidywania autora (przewaga w H1–H3) nie zostały potwierdzone: H1 i H3 – wynik przeciwny (klasa G), H2 – równoważność zamiast przewagi”. Opcjonalnie dodać zdanie o mocy H3.

### R6-09 (istotna) – E3: „`K-LTG` na F-MNIST wyraźnie gorsze od bloków referencyjnych” – nieprawda wobec dwóch z sześciu

**Lokalizacja:** część 5, E3, drugi podpunkt.

**Dowód:**
- `K-LTG` − `K-L` (F-MNIST): +0.42 [+0.22, +0.61].
- `K-LTG` − `K-LLT`: +0.59 [+0.41, +0.77].
- `K-LTG` jest gorsze od `K-R`, `K-G`, `K-T` i `K-GGT`, ale lepsze od `K-L` i `K-LLT`.
- Słowo „wyraźnie” jest ocenne.
- „Żadna nie jest lepsza od `K-GGT` na F-MNIST” jest dosłownie fałszywe dla średniej: `K-TGL` − `K-GGT` = +0.01 [−0.13, +0.15].

**Poprawka:**
- Napisać: „`K-LTG` na F-MNIST poniżej `K-R`, `K-G`, `K-T`, `K-GGT` (np. vs `K-R` −0.89 [−1.06, −0.71]), powyżej `K-L` i `K-LLT`”.
- Drugie zdanie zmienić na: „żadna inna kolejność nie ma średniej istotnie wyższej od `K-GGT` (95% CI)”.

### R6-10 (drobna, na granicy istotnej) – zdanie o twierdzeniach o rozkładzie danych dla H2 jest za szerokie

**Lokalizacja:** część 4, ostatnie zdanie.

**Dowód:**
- W (a) H2 ma łącznie RÓWNOWAŻNOŚĆ+, ale per zbiór MNIST RÓWNOWAŻNOŚĆ+, F-MNIST **RÓWNOWAŻNOŚĆ** (direction = 0).
- Na poziomie łącznym werdykt jest zgodny z głównym, więc reguła §7.8(a) jest spełniona.
- Część „+” i kierunek „−” na F-MNIST nie są jednak potwierdzone na poziomie rozkładu.

**Poprawka:** „Twierdzenia o rozkładzie danych dopuszczalne dla H3 (klasa G, `P-R` i `P-G`) oraz dla H2 w zakresie równoważności (±S). Istotna różnica w H2 na F-MNIST nie utrzymuje się w (a). Dla H1 niedopuszczalne”.

### R6-11 (drobna) – dopisek H3 na pierwszej stronie nie jest dosłowny i może mylić

**Lokalizacja:** pierwsza strona, punkt H3.

**Dowód:**
- Zamiast wymaganego zdania jest jego parafraza („…mniejsza niż przyjęty próg S na F-MNIST; jej praktyczne znaczenie na obu zbiorach łącznie nie zostało wykazane”).
- Na F-MNIST `P-R`: Δ = −1.268 [−1.411, −1.125] przy S = 1.35. Przedział obejmuje −S. Różnica nie jest więc „mniejsza niż S” w żadnym mocnym sensie. GORSZOŚĆ<S znaczy tylko, że przekroczenia S nie wykazano.
- „Różnica > S” przy MNIST należy rozumieć jako wykazane inf(S).

**Poprawka:**
- Dać dosłowny dopisek §7.4.
- Dodać zdanie opisowe: „na F-MNIST estymata `P-R` (−1.27) jest bliska S = 1.35, a PU obejmuje −S”.
- Zamiast „różnica > S” napisać: „Na MNIST GORSZOŚĆ (inf(S): |Δ| > S wykazane dla `P-R`, `P-G`)”.

### R6-12 (drobna) – mechanizmy w części eksploracyjnej słabo uzasadnione lub sprzeczne z własnymi danymi

**Lokalizacja:** część 5, E1, E3 (trzeci podpunkt) i E7.

**Dowód:**
- E1: hipoteza, że głębszy blok „kompensuje problem ograniczonej dobroci tanh”. `K-T` ma wyjście tanh, więc nadal G ≤ w (§13.5). Ograniczenie dobroci zostaje, a mechanizm musi być inny, np. optymalizacja lub inicjalizacja.
- E3: hipoteza „bank szkodzi… zgodnie z obserwacją Akkusa o sumie kwadratów przy gęstych aktywacjach”. Płaskie GELU (gęste aktywacje) wypada dobrze: `P-G` 97.19 / 86.10, drugie miejsce wśród ramion P, wyraźnie powyżej `P-BANK`. Słabe jest `P-L` (95.11 / 83.60). Ten kontrprzykład trzeba podać.
- E7: zdanie „obecność tanh na wyjściu bloku sama w sobie nie wyjaśnia różnic” jest wnioskiem przyczynowym, a nie hipotezą.
- E7: brakuje pozostałych kontrastów z §7.7. Są w JSON: LGT − GLT = −0.02 [−0.09, +0.05] / −0.02 [−0.19, +0.15] oraz LGT vs `K-T`, `K-LLT`, `K-GGT`.
- E7: opis „bloki z wyjściem tanh − pozostałe” nie mówi, że chodzi o średnią(LGT, GLT) − średnią(4 pozostałych permutacji).

**Poprawka:**
- Przeformułować E1 i E3, w E3 z kontrprzykładem `P-G`.
- W E7 dodać brakujące kontrasty, sformułować wniosek jako hipotezę i doprecyzować definicję.

### R6-13 (drobna) – drobne niezgodności liczb i opisu źródeł

- E1, T, F-MNIST: w raporcie +3.15 [+3.03, **+3.27**], w JSON i w moim przeliczeniu [+3.03, **+3.26**] (3.2648).
- Nagłówek raportu mówi, że liczby są „przepisane bez zmian z `WYNIKI_KONCOWE.json`”. Część liczb policzył jednak autor poza `final.py`: różnice E9 (K-LGT − najlepszy, P-BANK − P-R), zliczenia limitu i „epoki 0” z `runs/*.json`, tabela E10 z `e10_wyniki.json`. Trzeba to oznaczyć.
- „Obserwacja nieplanowana”: „epoka 0” to indeks od 0 (`train.py:76`), czyli najlepsza była pierwsza epoka. Należy to dopisać, bo „epoka 0” brzmi jak warstwa nieuczona.
- E6: lista „najwyższe” miesza kolejność zbiorów. Na F-MNIST `K-GGT` (87.32) jest drugie, przed `K-TLG` (87.21). Lepiej podać dwie osobne listy albo ranking per zbiór.
- E3: przewaga `K-BANK` jako „najwyższej średniej” wynosi tylko 0.02 p.p. (vs `K-GGT` na MNIST, vs `K-TGL` na F-MNIST). Warto to podać, żeby zdanie nie brzmiało jak wynik główny.

### R6-14 (drobna) – identyfikowalność i środowisko

- Status badania cytuje commit hashy `dc48e63` (gałąź `ff-od-zera-public`). `ODSTEPSTWA.md` cytuje `4b26d2c` (gałąź `master`). Oba commity mają tę samą treść i czas 19:26:02 +0200. Należy podać oba i wyjaśnić, że gałąź publiczna to lustro.
- Wersje: rekordy biegów zawierają Python, torch, CUDA, GPU, sterownik i commit kodu, ale nie numpy. Źródło „numpy 2.1.2 / 2.1.3” nie wynika z rekordów, więc trzeba je podać.
- Raport nie podaje commitu kodu Etapu C (`8df4f0f`) ani tego, że trzy instancje miały **różne sterowniki hosta** (580.159.03, 580.178.04, 570.133.20; po ok. 1/3 biegów). Nie narusza to §8, ale powinno być w wierszu „GPU, obraz, wersje”.
- Reguła brzegowa: szablon wymaga podania osi i liczby punktów. Osie to **wd i c** na obu zbiorach (`plany/B2.json`), 456 biegów = 152 punkty × 3 ziarna. Raport podaje tylko liczbę biegów.
- Koszt: dopisać źródło (saldo Vast.ai przed D1 i po zakończeniu) oraz to, że E10 i lokalna ewaluacja CPU nie są wliczone.
- §7.8(c) „nie dotyczy (brak powtórek)”: dopisać, że biegi przerwane przy przełączeniu instancji (ODSTEPSTWA, poz. 3) uruchomiono od zera zgodnie z §8 i nie są powtórkami w rozumieniu §5.6 pkt 4.
- Odchylenie w E9 od definicji „najlepszego ramienia” (wybór według testu) – jeśli zostanie po poprawce R6-04, wpisać je do `ODSTEPSTWA.md`.

### R6-15 (drobna) – część 3 i 4: brak przedziałów Hodgesa–Lehmanna i opisu (b)

- §7.6 i §7.8(b) przewidują dla wariantu Wilcoxona przedział Hodgesa–Lehmanna. Raport podaje tylko kategorie.
- Ponieważ NaN = 0, Wilcoxon pełni tylko rolę analizy wrażliwości. Wystarczy to zdanie, z przedziałami albo uzasadnieniem ich pominięcia (kod `final.py` ich nie liczy).

---

## Werdykt

- **Rdzeń konfirmacyjny jest rzetelny:**
  - liczby w częściach 2–4 zgadzają się z danymi;
  - kategorie, łączenie i zdania z macierzy są dobrane poprawnie;
  - słów zakazanych nie ma.
- **Raportu nie należy pokazywać w obecnej postaci**, z dwóch powodów.
  1. **R6-01 (krytyczna):**
     - na pierwszej stronie i w obowiązkowym dopisku `K-R` jest nazwany „istotnie lepszym od LGT”;
     - według procedury Holma z §7.3 to twierdzenie jest fałszywe;
     - poprawna lista to wyłącznie `K-GGT`.
  2. **R6-07:** brakuje elementów obowiązkowych (E5, Rysunek 2, wymagany opis Rysunku 1). Raport można pokazać najwyżej jako wersję roboczą, z wyraźną listą braków na pierwszej stronie.
- Uwagi istotne R6-02 – R6-06, R6-08 i R6-09 trzeba poprawić przed wersją końcową:
  - kierunek H2;
  - definicja limitu;
  - E9 według definicji z protokołu, z CI;
  - kontekst E2;
  - E10;
  - przewidywania na pierwszej stronie;
  - E3.
  Są to poprawki redakcyjne lub krótkie przeliczenia eksploracyjne. Żadna nie zmienia kategorii konfirmacyjnych.
- **Po poprawkach R6-01 – R6-09** (drobne wedle uznania) raport będzie gotowy do pokazania.
