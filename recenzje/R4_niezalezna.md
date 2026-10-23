# Recenzja R4 (niezależna) – PROTOKOL_v4.md

**Recenzent:** R4, niezależny. Widzi protokół po raz pierwszy i nie był zaangażowany w rundy R1–R3.
**Data:** 23.10.2026
**Przedmiot:** `PROTOKOL_v4.md` i `DECYZJE_AUTORA.md` (D1). Kontekst: `00_AUDYT_BLEDOW.md`, dane pilotowe `eksperyment_kontrola_wymiaru_ff/wyniki_kontrola_wymiaru.json` z kodem modeli.

**Kolejność pracy.** Najpierw samodzielna lektura protokołu i danych pilotowych. Dopiero potem historia R1–R3 i O1–O3, sprawdzona pod kątem powtórzeń.

**Obliczenia własne:**
- skrypt `/tmp/claude-1000/r4_power.py`: pełna implementacja reguł §7.3 (warunki elementarne, Holm na α'/2, kolejność 1–7), łączenie zbiorów według §7.4, Monte Carlo z 20 000 powtórzeń;
- przeliczenia z pilota wykonane bezpośrednio na pliku JSON.

---

## 1. Werdykt

**POWAŻNE POPRAWKI**

Logika decyzyjna §7.3–7.4 jest poprawna. Kontrolę błędu I rodzaju przeliczyłem niezależnie i zgadza się z R3.

Zostały jednak dwa problemy, które przeoczono w trzech rundach:
1. Reguła wyznaczania N opiera się na wariancji niewłaściwego kontrastu. Własne dane pilotowe autora pokazują, że ten kontrast ma najmniejszą wariancję ze wszystkich.
2. Część zdań w wiążącej macierzy interpretacji przypisuje efekty „różnorodności” lub „mieszaniu”. Projekt tego nie identyfikuje, a §1 wprost to wyklucza.

Do tego dochodzą:
- nieujawnione ponowne użycie zbioru testowego MNIST z pilotów (drobne, bo obciążenie jest znikome);
- kilka luk wykonawczych specyficznych dla Vast.ai.

Wszystkie poprawki są tanie: kilka dodatkowych ramion w mini-pilocie (rzędu centów), zmiany tekstu i opcjonalnie jedna dodatkowa ewaluacja.

**Podsumowanie:** 0 KRYTYCZNYCH, 2 POWAŻNE, 9 DROBNYCH.

---

## 2. Tabela zarzutów

| ID | Waga | Sekcja | Skrót |
|---|---|---|---|
| R4-01 | POWAŻNY | §7.5, §9.3 | σ_d w regule N pochodzi z kontrastu międzytopologicznego `K-LGT − P-R`, którego nie używa żaden test. W pilocie ten kontrast ma SD 0.18, a kontrasty wewnątrzgrupowe 0.28–0.34. Czynnik 1.29 nie pokrywa różnicy ~1.9×. |
| R4-02 | DROBNY | §4.1, §7.1, §7.8, §13 | Oficjalny zbiór testowy MNIST był używany w pilotach (audyt K9), po których wybrano LGT jako ramię badane i kierunek H1. Estymand jest warunkowy względem tego zbioru. Obciążenie jest znikome, ale protokół tego nie ujawnia. |
| R4-03 | POWAŻNY | §7.4 (macierz H1×H2), §9.4 | Komórki P×G, G×P, G×R i R×P zawierają twierdzenia nietestowane albo uogólniające („różnorodność pomaga/nie pomaga”, „mieszanie szkodzi”, „inne kolejności tracą względem monokultur”). Kłóci się to z §1 i z zakłóceniem §13.5. Szablon wiąże raport z tymi zdaniami. |
| R4-04 | DROBNY | §8, D1 pkt 4 | Wyniki są kopiowane lokalnie dopiero na końcu etapu. Utrata instancji w Vast.ai kasuje cały etap, a przy limicie 38 USD i regule „wstrzymanie” może zatrzymać badanie. |
| R4-05 | DROBNY | §5.7, §8, §9.1.6, §9.1.9 | Determinizm GPU. Test 9.1.9 (bitowa zgodność buforowania z przeliczaniem) może nie przejść przy poprawnej implementacji. Nowa instancja w trakcie etapu nie powtarza 9.1.6. Nie określono urządzenia generatorów. |
| R4-06 | DROBNY | §5.6 p. 4, §8 | Pozostałość R2-04. „Błąd kodu” nie ma definicji operacyjnej, a metryki walidacyjne Etapu C są jawne. Pod pozorem poprawki może przejść zmiana specyfikacji. |
| R4-07 | DROBNY | §5.2 | Reguła E_max działa dopiero po Etapie B, więc strojenie A/B przy aktywnym limicie może różnicowo krzywdzić wolno zbiegające ramiona. Nie wiadomo, której warstwy dotyczy „epoka najlepsza”. |
| R4-08 | DROBNY | §7.6 | NaN w warstwie 2 przy dobrej warstwie 1 daje według litery reguły „wyjście losowe” (~10%) zamiast klasyfikatora z G₁. |
| R4-09 | DROBNY | §7.5, §7.4 | SESOI na Fashion-MNIST (10% z ok. 12–15% błędu, czyli 1.2–1.5 p.p.) sprawia, że realny, ale mały efekt trafia tam do klasy R (RÓWNOWAŻNOŚĆ+). Zestawienie MNIST P + F-MNIST R daje NIESPÓJNY-CZĘŚCIOWY, więc F-MNIST praktycznie blokuje łączne P i G. Trzeba to ujawnić i przemyśleć. |
| R4-10 | DROBNY | §4.1, §5.7 | Brak hashy plików danych i opisu ich dostarczenia na instancję. |
| R4-11 | DROBNY | §7.2, §7.4, §9.3, §11, §12, D1 | Niespójności: liczba biegów E2, odwołania D1 → §11.2, opcja „tańszy GPU” sprzeczna z §9.3 i D1, nieaktualne godziny CPU, punkt 5 mini-pilota bez miejsca w §7.5, niewprowadzone zdanie z R3 o α, kierunki RÓWNOWAŻNOŚCI+, brak reguły na przerwanie badania. |

---

## 3. Szczegóły

### R4-01 [POWAŻNY] σ_d z niewłaściwego kontrastu (§7.5, §9.3)

**Problem.** Reguła N (§7.5) używa σ_d = max(1.29 × SD różnic `K-LGT − P-R`, 0.5·S). Żaden test konfirmacyjny nie porównuje jednak `K-` z `P-`:
- H1 i H2 to kontrasty wewnątrz topologii K,
- H3 to kontrast wewnątrz topologii P.

Wariancja kontrastu zależy od korelacji obu ramion po ziarnie, a ta jest różna dla par wewnątrz- i międzytopologicznych.

**Dowód z pilota.** Dane: `wyniki_kontrola_wymiaru.json`, N = 10, MNIST, test, stały podział. Mapowanie ramion sprawdziłem w `model_*.py`:
- C = blok L→G→T (≈ `K-LGT`),
- D = blok GELU³ (≈ `K-G`),
- E = płaska ReLU 250 (≈ `P-R`),
- B = bank (≈ `P-BANK`).

| Kontrast pilota | Odpowiednik | Używany w | SD różnic |
|---|---|---|---|
| C − E | `K-LGT − P-R` | **reguła N** | **0.177** |
| C − D | `K-LGT − K-G` | H1 | 0.337 |
| B − E | `P-BANK − P-R` | H3 | 0.284 |
| D − E | `K-G − P-R` | – | 0.330 |

Kontrast wybrany do reguły N ma **najmniejsze** SD ze wszystkich. Kontrasty hipotez mają 1.6–1.9 razy większe. Czynnik 1.29 (górna 80% granica) miał chronić przed niepewnością estymatora właściwej wielkości. Nie chroni przed estymowaniem niewłaściwej wielkości.

**Zastrzeżenia uczciwości:**
- W pilocie θ różniło się między ramionami (K1: C 62.5, D 125, B 100, E 125). Część wariancji C−D może wynikać z innego θ. W Etapie C konfiguracje też będą różne dla każdego ramienia (strojenie), więc ten mechanizm zostaje.
- Mini-pilot mierzy SD na walidacji, a zbiór walidacyjny jest inny dla każdego ziarna. Do SD dochodzi więc składowa próbkowania przykładów, której w teście na stałym zbiorze nie ma. Przy rozbieżności predykcji ok. 2–3% to ok. √(0.025/10 000) ≈ 0.16 p.p. Daje to 0.177 → ~0.24, a po × 1.29 → ~0.31. Efekt częściowo kompensuje problem, ale tylko przypadkiem i w nieznanej proporcji.
- Losowy podział 50k/10k dla każdego ziarna doda wariancji we wszystkich kontrastach, której pilot nie zawierał.

**Skala (moja symulacja potoku §7.3–7.4).** Założenia: m = 5, ρ = 0.5. MNIST: S = 0.30 (10% z ~3% błędu C). Fashion-MNIST: S_F = 1.2 i σ_F = 0.5. **Wartości dla F-MNIST to założenia, bo nie ma żadnych danych pilotowych FF na tym zbiorze.**

| σ_d MNIST | S/σ | N=30: P / R | N=40: P / R | N=50: P / R | N=60: P / R |
|---|---|---|---|---|---|
| 0.23 (1.29 × 0.177) | 1.30 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 0.30 | 1.00 | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 0.34 (C−D z pilota) | 0.88 | 0.99 / 0.95 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 0.43 (1.29 × 0.337) | 0.70 | 0.89 / **0.60** | 0.97 / 0.88 | 0.99 / 0.97 | 1.00 / 0.99 |
| 0.50 | 0.60 | 0.77 / **0.25** | 0.89 / 0.61 | 0.96 / 0.83 | 0.98 / 0.94 |

Wariant niejednorodny: 4 komparatory z σ = 0.23 i jeden (np. `K-T`) z 0.46. Moc klasy R wynosi wtedy 0.81 przy N = 30 i 0.94 przy N = 40.

**Scenariusz błędu.** Mini-pilot daje SD(`K-LGT − P-R`) ≈ 0.2–0.25, więc reguła wybiera N = 30. Prawdziwe σ kontrastów H1/H2 w Etapie C wynosi 0.4–0.5. Moc celu (ii) spada z deklarowanych ≥ 0.80 do 0.25–0.60. Twierdzenie „kolejność obojętna” (komórki z H2 = R) prawie na pewno skończy się NIEROZSTRZYGNIĘTYM. Nie unieważnia to wniosków, ale może zmarnować badanie, a protokół deklaruje moc, której nie ma.

**Poprawka (koszt rzędu 40–80 biegów mini-pilota, centy):**
1. Do mini-pilota dodać ramiona wewnątrzgrupowe, minimum `K-G` i `K-GLT`; lepiej także `K-T` i `P-BANK` lub `P-T`.
2. σ_d per zbiór = 1.29 × **max** SD po wszystkich zmierzonych kontrastach wewnątrzgrupowych (`K-LGT − K-x`, `P-BANK − P-x` albo `P-x − P-R`).
3. Z tych samych danych zmierzyć ρ i wpisać do symulacji max(ρ̂, 0.5) albo ρ̂ z uzasadnieniem.
4. Usunąć `K-LGT − P-R` jako podstawę reguły N. Można go zostawić jako informację.

### R4-02 [DROBNY] Ponowne użycie zbioru testowego MNIST (§4.1, §7.1, §7.8, §13)

**Problem.** Według audytu K9 w `eksperyment_kontrola_wymiaru_ff` wszystkie ewaluacje odbywały się na oficjalnym teście MNIST (`run_experiment.py:55`, `datasets.MNIST(train=False)`). Sam projekt bloku L→G→T (LeakyReLU 0.5) **poprzedza** te wyniki:
- `model_laminar_cascade.py` ma datę 17:04;
- Etap 1 ruszył o 17:25 (chronologia audytu).

Po wynikach zapadły natomiast dwie decyzje: wybór LGT jako ramienia badanego i kierunek przewidywania H1. §7.1 ustala estymand warunkowy względem tego samego zbioru testowego, a protokół nie ujawnia, że ten zbiór był już oglądany.

**Skala obciążenia: znikoma.**
- H2: żadna z pięciu innych kolejności nie była w pilocie, więc nie ma składowej selekcji.
- H1: w pilocie był tylko odpowiednik `K-G` (D). C wygrało z D o 0.93 p.p. przy SE składowej testowej ≈ 0.1–0.2 p.p. (R1-07). Przy takim marginesie warunkowanie na „C wygrało” prawie nic nie mówi o składowej specyficznej dla testu, więc oczekiwane obciążenie jest bliskie 0.
- H3: pilot dał B < E, czyli obciążenie, jeśli jakieś jest, działa przeciw przewidywaniu autora.
- Fashion-MNIST nie był używany w pilotach (grep po katalogach pilotów), a łączenie IUT wymaga zgodności obu zbiorów.

**Poprawka:**
1. Jedno zdanie w §13 (i §4.1): test MNIST był używany w pilotach, po których wybrano LGT i kierunek H1. Projekt bloku powstał przed tymi wynikami.
2. Opcjonalnie, jako analiza wrażliwości §7.8(d): ewaluacja tych samych checkpointów Etapu C na 50 000 „zaginionych” cyfr testowych QMNIST (Yadav i Bottou, 2019, *Cold Case: The Lost MNIST Digits*, NeurIPS 32). Te próbki nigdy nie były używane w serii. W torchvision: `QMNIST(what='test50k')`; do sprawdzenia przy implementacji, z hashem pliku. Koszt to tylko inferencja. Wymaga dopisania QMNIST do testu 9.1.5. Nie jest to żądanie nowego zbioru, bo bez tej analizy H1–H3 pozostają ważne.

### R4-03 [POWAŻNY] Zdania w macierzy §7.4 wychodzą poza identyfikację (§7.4, §9.4)

Szablon §9.4 i uwagi R3 („raport nie wychodzi poza zdania z macierzy”) czynią te zdania wnioskami konfirmacyjnymi. Kontrasty H1 nie są czyste:
- `K-R`, `K-L` i `K-G` różnią się od `K-LGT` jednocześnie heterogenicznością i typem nieliniowości wyjściowej (nieograniczona dobroć zamiast G ≤ 250, §13.5);
- tylko `K-T` ma to samo wyjście.

| Komórka | Obecne zdanie | Problem | Proponowane |
|---|---|---|---|
| P×G | „różnorodność pomaga, LGT nie jest najlepszą kolejnością” | „Różnorodność” w ogóle jest wykluczona przez §1. To, że lepsza kolejność > monokultury, nie jest testowane (E3). | „LGT lepszy od każdej monokultury; co najmniej jedna inna kolejność tych samych nieliniowości jest lepsza od LGT.” |
| G×P | „różnorodność nie pomaga w porównaniu z najlepszą monokulturą” | jw. | „LGT gorszy od co najmniej jednej monokultury i lepszy od każdej innej kolejności.” + obowiązkowe wskazanie, czy przegrana dotyczy monokultury z wyjściem nieograniczonym (§13.5, E7). |
| G×R | „mieszanie nieliniowości w bloku szkodzi” | Twierdzenie przyczynowe i ogólne. G<S (np. 0.1 p.p.) i równoważność kolejności w ±S nie wykazują, że *inne* kolejności są gorsze od tej monokultury. Przegrana z `K-G`/`K-R` może wynikać z ograniczonej dobroci tanh, a nie z mieszania. | „LGT gorszy od co najmniej jednej monokultury; pozostałe kolejności równoważne LGT w zakresie S.” + to samo wskazanie co wyżej. |
| R×P | „inne kolejności tracą względem monokultur, a LGT nie” | Kontrast „inne kolejności vs monokultury” nie jest testowany (E3). Przy S = 0.3 i różnicach < S może być fałszywy. | „LGT równoważny monokulturom (±S) i lepszy od każdej innej kolejności; porównanie innych kolejności z monokulturami – eksploracyjne (E3, E7).” |

**Poprawka.**
- Zastąpić te cztery zdania wersjami opisowymi jak wyżej.
- Dodać regułę ogólną: każda komórka z klasą G w H1 musi w raporcie podać, które komparatory przegrywają lub wygrywają i jaką mają nieliniowość wyjściową.
- Przypisanie mechanizmu („różnorodność”, „kolejność”, „wyjście tanh”) jest dozwolone wyłącznie w sekcji eksploracyjnej.

### R4-04 [DROBNY] Utrata instancji kasuje etap (§8, D1 pkt 4)

**Problem.** §8 i D1 przewidują kopiowanie wyników „przed zniszczeniem instancji”. Instancja on-demand o niezawodności ≥ 0.98 może jednak zniknąć (awaria hosta, wycofanie maszyny). Wtedy przepadają wszystkie ukończone biegi etapu, a nie tylko przerwane. Reguła „wznowienie przerwanych biegów od zera” zakłada, że ukończone przetrwają.

**Scenariusz.** Utrata hosta w 80% Etapu C to utrata ~80% kosztu C. Przy limicie 18 USD na C + E2 i 3 USD rezerwy kończy się to wstrzymaniem badania (§11.3).

**Poprawka.**
- Po każdym ukończonym biegu (lub co ≤ 30 min) synchronizacja `rsync` JSON-ów i checkpointów na maszynę lokalną, z weryfikacją hashy.
- Bieg uznaje się za ukończony dopiero po potwierdzonej kopii lokalnej.
- JSON-y z metrykami walidacyjnymi Etapu C trafiają do katalogu, którego autor nie otwiera do kroku 2 §5.6 (por. R4-06).

**Uwaga o budżecie.** Mój zgrubny szacunek obciążenia:
- Liczba biegów: A 512, B ≤ 672, C 1280–1920, E2 104, mini-pilot ≥ 40, razem ~2 600–3 250.
- Sieć ma ~0.26 M parametrów przy batchu 128, więc koszt na GPU wyznacza raczej narzut wywołań i Pythona niż FLOP-y. Rzędu 30–90 s na bieg daje 20–80 h procesu GPU, a przy kilku procesach na kartę ~5–20 h ściany.
- Przy typowych 0.1–0.3 USD/h za kartę ≥ 8 GB budżet 38 USD wydaje się wystarczający z dużym zapasem.

Rzeczywiste wąskie gardło to liczba rdzeni CPU instancji (1 wątek na proces). Mini-pilot powinien mierzyć przepustowość przy liczbie procesów ≤ liczba vCPU.

Pamięć: bufor warstwy 1 to 50 000 × 10 etykiet × 250 × 4 B ≈ 0.5 GB na proces, plus kontekst CUDA ~0.3–0.5 GB. Przy 8 GB VRAM daje to najwyżej ~6–8 procesów. Pomiar przepustowości „na danych losowych” (§9.3) musi uwzględniać bufor.

### R4-05 [DROBNY] Determinizm na GPU – testy, których definicja może się nie utrzymać (§5.7, §8, §9.1)

1. **Test 9.1.9** (buforowanie bitowo równe przeliczaniu). Na GPU wynik GEMM dla danego wiersza może zależeć od kształtu macierzy, czyli liczby wierszy w batchu: cuBLAS dobiera inny kernel lub podział K. Bufor liczony w dużych porcjach i przeliczanie w batchach po 128 mogą się różnić na ostatnim bicie przy poprawnej implementacji. Jeśli test nie przejdzie, autor zmienia go po fakcie, czyli wprowadza zmianę protokołu w trakcie.
   **Poprawka:** zdefiniować przed zamrożeniem jedną z dwóch opcji:
   - (a) bufor i ewaluacja liczone w tym samym, stałym podziale na porcje, a test bitowy dotyczy pary „trening z buforem” vs „trening z przeliczaniem w identycznych porcjach”;
   - (b) kryterium tolerancji: max |Δa| < 1e-5 i identyczne predykcje na walidacji 5k.
2. **Nowa instancja w trakcie etapu** (§8, „awaria instancji = nowa instancja tego samego typu”). Obraz Dockera ustala CUDA runtime, ale nie sterownik hosta. Test 9.1.6 trzeba powtarzać na każdej nowej instancji (koszt minut), z wynikiem w logu. Brak bitowej zgodności między hostami nie obciąża porównań, bo parowanie nie wymaga determinizmu, ale musi być odnotowany.
3. **Urządzenie generatorów.** `torch.Generator` na CPU i na CUDA daje różne sekwencje. Należy zapisać, że strumienie `split`, `order`, `neg` i `init` są generatorami CPU, a indeksy są przenoszone na GPU. Sposób mapowania krotki (s, k, warstwa, epoka) na ziarno również trzeba ustalić (np. SHA-256 krotki → 63 bity).

### R4-06 [DROBNY] „Błąd kodu” bez definicji operacyjnej (§5.6 p. 4) – częściowa niezgoda z zamknięciem R2-04

Reguła pokrycia z diffu zamknęła selektywne powtarzanie *ramion*, ale nie zamknęła selektywnego *definiowania poprawki*. Metryki walidacyjne Etapu C są w JSON-ach (§8), więc kontrasty H1–H3 da się policzyć przed testem.

„Poprawka błędu” może być w rzeczywistości zmianą specyfikacji, na przykład:
- inna inicjalizacja warstw tanh,
- inna obsługa nasycenia,
- inne eps.

Taka zmiana dotyka wielu ramion, ale nie jednakowo.

**Poprawka:**
- Błąd kodu = niezgodność implementacji z §3, §5 lub §5.7, wykazana testem, który nie przechodzi na kodzie pierwotnym i przechodzi na poprawionym. Test commitowany razem z diffem.
- Zmiany niewykazywalne w ten sposób nie uzasadniają powtórki.
- Dodatkowo metryki walidacyjne Etapu C zapisywać do osobnego pliku, którego nie otwiera się do kroku 2. Wczesne zatrzymanie działa wewnątrz runnera i tego nie potrzebuje.

### R4-07 [DROBNY] Reguła E_max po Etapie B i jej niejednoznaczność (§5.2)

**Problem.** Etapy A i B działają z E_max z mini-pilota, którego konfiguracja (lr 1e-3) nie jest strojona. Jeśli limit jest aktywny w A/B, strojenie premiuje konfiguracje szybko zbiegające. Ramiona wolniej zbiegające (np. tanh z małym lr) tracą przy wyborze konfiguracji, a podniesienie E_max dopiero w Etapie C tego nie cofa. Efekt może być różny dla różnych ramion.

Dodatkowo „najlepsza epoka” jest niejednoznaczna, bo każda warstwa ma własną.

**Poprawka:**
- To samo kryterium (> 25% biegów ramienia z najlepszą epoką ≥ E_max − 2 *w którejkolwiek warstwie*) sprawdzać także po Etapie A, przed B. Podniesienie obowiązuje wtedy od Etapu B.
- Reguła po B pozostaje bez zmian.
- Ująć to w budżecie scenariusza najgorszego.

### R4-08 [DROBNY] NaN w warstwie 2 (§7.6)

Litera reguły brzmi: „model z najlepszej dotychczasowej epoki; gdy takiej nie ma – wyjście losowe”. Dla NaN w 1. epoce warstwy 2 daje to ~10%, choć warstwa 1 jest wytrenowana. To niepotrzebnie wyzwala gałąź Wilcoxona i NIEROZSTRZYGNIĘTY (niestabilność).

**Poprawka:** „gdy warstwa 2 nie ma żadnej epoki bez NaN, klasyfikacja = argmax G₁”. Wyjście losowe tylko przy NaN w warstwie 1 przed pierwszą ukończoną epoką.

### R4-09 [DROBNY] SESOI na Fashion-MNIST i jego skutek dla łączenia zbiorów (§7.5, §7.4)

**Problem.** 10% z ok. 12–15% błędu FF na F-MNIST daje S_F ≈ 1.2–1.5 p.p. Nie mamy danych pilotowych FF na F-MNIST, więc typowej rozpiętości ramion nie znamy. Jeśli jednak różnice między ramionami są wyraźnie mniejsze niż S_F, warunek eq jest tam łatwy do spełnienia. Kategoria 3 (RÓWNOWAŻNOŚĆ+ = eq ∧ (sup(0) ∨ inf(0))) stoi przed kategoriami 4–5, więc realny, mały efekt na F-MNIST trafia do **klasy R**, a nie P ani G.

**Symulacja.** Δ_F = 0.5 p.p. wobec wszystkich 5 komparatorów, S_F = 1.2, σ_F = 0.5 (założenia), N = 30/40/60. P(klasa R na F-MNIST) = 1.00, P(klasa P) = 0.00.

**Skutek.** MNIST P + F-MNIST R daje NIESPÓJNY-CZĘŚCIOWY, czyli klasę N. Zgodny kierunkowo, mały efekt na obu zbiorach kończy się więc „nierozstrzygnięty”, bo F-MNIST w praktyce blokuje łączne werdykty P i G. Cel mocy (i) w §7.5 to obejmuje: przy Δ = S na każdym zbiorze klasa P jest osiągalna. Poza scenariuszem projektowym (Δ_F ≪ S_F, ale > 0) reguła daje jednak wynik mylący.

**Poprawka (do wyboru, przed zamrożeniem):**
- (a) zapisać jawnie w §7.4 i §13, że zestawienie P/R⁺ jest raportowane jako „ten sam kierunek, na jednym zbiorze różnica < S” i ma osobne zdanie w szablonie (nadal klasa N, ale bez sugestii sprzeczności);
- albo (b) łączyć zbiory po kierunku (sup(0) na obu → „P<S”), traktując RÓWNOWAŻNOŚĆ+ z kierunkiem dodatnim jako zgodną z P<S.

Wariant (a) nie zmienia kontroli błędu.

### R4-10 [DROBNY] Pochodzenie danych (§4.1, §5.7)

Brakuje:
- hashy SHA-256 plików MNIST i F-MNIST (oraz QMNIST, jeśli R4-02 zostanie przyjęte);
- sposobu dostarczenia danych na instancję. Pobieranie przez torchvision z serwerów źródłowych bywa zawodne i może podmienić mirror.

**Poprawka:** dane w obrazie Dockera lub kopiowane `scp`, hashe w `SHA256SUMS` (§10.1), weryfikowane przez runner przy starcie.

### R4-11 [DROBNY] Spójność dokumentu

1. §11.2: „E2: 20 biegów BP”. Faktycznie jest 2 ramiona × 10 ziaren × 2 zbiory = **40** biegów końcowych plus strojenie 2 × 16 punktów × 2 zbiory = **64**, razem 104.
2. D1 pkt 2 odsyła do „§11.2 (pkt 2 i 3)”, a redukcje są w §11.3 (pkt 1–3, z pominięciem E2). Ujednolicić: czy E2 wchodzi do progu 33 USD (D1 mówi „Etapy A–C”)?
3. §11.3 pkt 3 („tańszy typ GPU, jeśli pomiar wykaże niższy koszt”) jest sprzeczny z §9.3 („ten sam co w Etapach A–C”) i z D1 pkt 1 (najtańsza oferta wybierana już przed mini-pilotem). Mini-pilot mierzy tylko jeden typ. Usunąć albo opisać procedurę pomiaru drugiego typu przed zamrożeniem.
4. Pod-limity D1 (5/12/18/3 USD): nie zapisano, co się dzieje przy przekroczeniu pod-limitu przy łącznym wydatku poniżej 38 USD.
5. §11.2: odniesienie „3 rdzenie i5-3450, 200–480 h CPU” jest nieaktualne po D1. Oznaczyć jako historyczne albo usunąć.
6. §9.3 pkt 5 („składowa wariancji z próbkowania przykładów → raport w §7.5”): §7.5 nie ma na to miejsca. Dodać wiersz.
7. R3 zalecał dopisanie w §7.2 zdania, że α = 0.05 jest kontrolowane dla każdego typu twierdzenia (P, R, G) osobno, a łączne prawdopodobieństwo fałszywego P lub R w jednej hipotezie może sięgać 1.5α'. Tego nie wprowadzono, a §12.4 tego nie odnotowuje.
8. §7.4: przy RÓWNOWAŻNOŚCI+ z przeciwnymi kierunkami na dwóch zbiorach dopisek musi podawać oba kierunki. Obecnie jest tylko „z kierunkiem”.
9. Brak reguły na przerwanie badania po zamrożeniu (np. brak środków po „wstrzymaniu”). Dopisać: ewaluacja testowa częściowego Etapu C jest zabroniona, a raport opisuje badanie jako nieukończone.

**Sprawdzone i poprawne:**
- liczby parametrów §3.1 (259 000; 195 687 + 62 565 = 258 252; −0.29% per warstwa);
- BP 261 510 i 260 762;
- podziały BANK (84/83/83, 55/54/54, 35/34/34);
- czynniki 1.29 i 1.47 (√(9/χ²₀.₂₀,₉) = 1.293, √(9/χ²₀.₁₀,₉) = 1.469);
- liczby kompletów A (32) i B (18 + ≤ 24);
- wzajemne wykluczanie kategorii 1–7;
- kontrola błędu I rodzaju, w pełni zgodna z R3 §3.

---

## 4. Rozstrzygnięcia poprzedników, z którymi się nie zgadzam

1. **R2-09, „uwaga pozytywna”: σ_d z `K-LGT − P-R` „prawdopodobnie zawyża σ_d dla kontrastów wewnątrz grup, więc działa zachowawczo”.** Dane pilotowe, które R1 przytoczył w swoim §4 (weryfikacja twierdzeń: C−D 0.34, B−E 0.28, C−E 0.18), pokazują coś odwrotnego. Na tym założeniu opierają się zamknięcia R3-01 i R3-02, a R3 w §4 pkt 3 wprost użył wartości 0.18. R3-01 dotyczył niepewności *właściwej* wielkości. Tu źle wybrana jest sama wielkość, dlatego daję wagę POWAŻNY (R4-01).
2. **R2-04 / R1-08 uznane za USUNIĘTE.** Zgadzam się, że selektywne powtarzanie ramion jest zamknięte. Otwarte zostaje selektywne definiowanie „poprawki” przy jawnej walidacji (R4-06, drobny).
3. **R2-03 (komórka R×P) uznane za USUNIĘTE.** Interpretacja nadal zawiera nietestowane twierdzenie „inne kolejności tracą względem monokultur” (R4-03).
4. **R1-07 (stały zbiór testowy) uznane za zamknięte.** Odparcie dotyczyło wariancji i jest rozsądne. Ponowne użycie testu MNIST nie było rozważane. Skutek jest znikomy, ale wymaga ujawnienia (R4-02, drobny).

Pozostałe rozstrzygnięcia (R1-01…R1-21, R2-01…R2-10, R3-01…R3-04) uważam za trafne.

---

## 5. Czy protokół jest gotowy do zamrożenia?

**Nie, nie w obecnym brzmieniu.**

Dwa zarzuty POWAŻNE wymagają zmian przed zamrożeniem:
- R4-01: zmiana składu mini-pilota i podstawy reguły N; musi nastąpić przed uruchomieniem mini-pilota;
- R4-03: przepisanie czterech komórek macierzy.

Obie zmiany są tanie (centy obliczeń, zmiany tekstu) i nie zmieniają architektury badania. Nie wymagają pełnej kolejnej rundy. Wystarczy ukierunkowana weryfikacja, czy:
1. mini-pilot zawiera ramiona wewnątrzgrupowe, a σ_d = 1.29 × max SD po nich;
2. komórki P×G, G×P, G×R i R×P mają brzmienie opisowe.

Zarzuty DROBNE (R4-02, R4-04 do R4-11) autor może wprowadzić bezpośrednio do wersji zamrażanej. Szczególnie zalecam:
- R4-04: synchronizacja po każdym biegu;
- R4-05 pkt 1: kryterium testu 9.1.9 zdefiniowane przed zamrożeniem;
- R4-09: decyzja (a) lub (b).

Bez R4-04 i R4-05 pkt 1 protokół może okazać się niewykonalny dokładnie tak, jak jest napisany.

---

**Źródła zweryfikowane w sieci:**
- Yadav, C., & Bottou, L. (2019). Cold Case: The Lost MNIST Digits. *NeurIPS 32*: https://papers.nips.cc/paper/9500-cold-case-the-lost-mnist-digits ; arXiv:1905.10498.
