# Audyt błędów w eksperymentach Forward-Forward z 17.10.2026

**Zakres:** `eksperyment_sca_ff/`, `eksperyment_laminar_ff/`, `eksperyment_kontrola_wymiaru_ff/`
**Data audytu:** 23.10.2026
**Źródła:** pliki w repozytorium (`git log`), logi Antigravity (rozmowa `3a9149f5-064b-4af1-9e9c-40f9bee5ac55`, subagent `e0bd6d2b-19ff-42af-9334-8ff3aebc5cd0`).

Każdy błąd ma identyfikator. W protokole `PROTOKOL_v*.md` tabela identyfikowalności (§12) wskazuje, gdzie dany błąd został usunięty.

Waga: **K** – krytyczny (unieważnia wniosek konfirmacyjny), **P** – poważny (recenzent by odrzucił), **D** – drobny.

---

## Chronologia (17.10.2026, czas lokalny)

| Godz. | Zdarzenie |
|---|---|
| 11:45 | Prerejestracja SCA-FF v1.0 (15 epok, 3 ziarna) |
| 12:08–12:26 | Bieg SCA-FF: 3 epoki, 2 ziarna; raport |
| 12:58–13:01 | Protokół, modele i runner Laminar-FF |
| 13:01:44 | Subagent uruchamia `run_experiment.py` (Laminar-FF) |
| ~13:08 | Bieg przerwany przez użytkownika (obciążenie podczas wykładu). **Brak wyników** – w folderze nie ma żadnego pliku wynikowego |
| ~15:58 | Recenzja (Claude) wykrywa złożenie Linear→Linear w kaskadzie |
| 16:19–17:09 | Prerejestracja `REG-FF-2026-DIM-01` v1.0 → v1.2 (5 commitów w ~50 min), modele |
| 17:19 | Commit `68a1ff6`: zmiana progów testów w runnerze po zamrożeniu protokołu (przed biegiem) |
| 17:25–22:51 | Etap 1 (N=10, 6 modeli, 15 epok) |
| 22:53 | `RAPORT_WYNIKOW_ETAP1.md` |

---

## A. `eksperyment_sca_ff`

| ID | Waga | Błąd | Dowód |
|---|---|---|---|
| S1 | K | Protokół: 15 epok, 3 ziarna `[42,1337,2024]`. Wykonano 3 epoki i 2 ziarna. Odstępstwo nie jest opisane w raporcie jako odstępstwo. | `PROTOKOL_BADANIA_PREREJESTRACJA.md` §4; `RAPORT_WYNIKOW.md` nagłówek |
| S2 | K | Próg H2 dla baseline'u: protokół 10%, kod 5% (`baseline_dead > 5.0`). Mimo to raport podaje warunek „> 10.0%”. | `run_experiment.py`, `generate_markdown_report` |
| S3 | P | H3 wymaga istotności p < 0.05. Żaden test nie został wykonany. Decyzja zapadła przez porównanie średnich z n=2. | `run_experiment.py` (`h3_pass = sca_acc > static_acc`) |
| S4 | P | Generator raportu ma wpisane na sztywno wnioski („korzyść … z ich dynamicznego dostosowywania”), niezależne od wyniku. Opublikowany raport został potem ręcznie przepisany. | `generate_markdown_report`, sekcja 5 |
| S5 | P | Silne niedotrenowanie (3 epoki). Baseline ReLU-FF uzyskał 88.6%. Przy 15 epokach ten sam typ modelu ma 95.8% (kontrola_wymiaru, Model A). Wnioski o „wąskim gardle ReLU” są artefaktem. | porównanie raportów |
| S6 | P | H1 porównuje modele o różnych budżetach parametrów (1.786M / 1.036M / 0.259M). Wymiar ukryty jest myląco utożsamiany z pojemnością. | tabela 1 raportu |
| S7 | P | Próg θ = d/2 dla wszystkich aktywacji, bez względu na ich zakres (tanh ∈ [−1,1]). Tworzy to zmienną zakłócającą. | `model_*.py` |
| S8 | D | SD liczone `np.std` (ddof=0) przy n=2. | `generate_markdown_report` |
| S9 | D | Definicja martwego neuronu E[a²] < 1e-4 nie jest niezmiennicza względem skali. Dla jednostek liniowych i tanh jest trywialnie spełniona jako 0%. | `evaluate_model` |
| S10 | D | Język: „jednoznacznie dowodzi”, „bezprecedensowo efektywny”. | `RAPORT_WYNIKOW.md` §5 |

## B. `eksperyment_laminar_ff`

| ID | Waga | Błąd | Dowód |
|---|---|---|---|
| L1 | K | Podwarstwa „liniowa” (`sub_lin`) wchodzi wprost w `sub_gelu` (Linear). Złożenie W₂W₁ to jedno odwzorowanie afiniczne o ograniczonym rzędzie, a nie „monokultura liniowa”. | `model_laminar_cascade.py` |
| L2 | K | Kaskada ma 3 podwarstwy z lokalnym backpropem wewnątrz makrowarstwy, a bank ma 1 warstwę. Głębokość jest pomieszana z polimorfizmem. Model C przestaje być czystym FF warstwa-po-warstwie. | §3 protokołu |
| L3 | P | Bieg uruchomiony 13:01:44, przerwany ok. 13:08, **brak wyników**. Folder pozostał jako „zamrożony” bez adnotacji o przerwaniu. | logi Antigravity; brak plików wynikowych |
| L4 | P | H4 (czas treningu, GPU, „branch divergence”) uzasadniona argumentami o GPU, a mierzona na współdzielonym CPU z `nice`. | §1.4, §2 H4 |
| L5 | P | θ = 125 dla wyjścia tanh wymaga średnio \|a\| > 0.707 (półnasycenie). Ten sam θ dla banku i kaskady przy różnych zakresach wyjść. | §3 |
| L6 | D | Mapowanie na warstwy kory (IV = liniowa, V/VI = tanh) przedstawione jako „ścisła formalizacja”, bez danych. | §1.3 |
| L7 | D | Ziarna `[42, 1337]` to te same, co w pilocie SCA. | §4.2 |

## C. `eksperyment_kontrola_wymiaru_ff`

| ID | Waga | Błąd | Dowód |
|---|---|---|---|
| K1 | K | Różne θ w każdym ramieniu (A 500, B 100, C 62.5, D 125, E 125), ustalone heurystycznie i bez równego strojenia. Kontrasty C−D i C−B mieszają architekturę z doborem θ. | `model_*.py`, protokół §3 |
| K2 | K | Prerejestrowany wariant D z θ = 62.5 (protokół §3.6) **nie został uruchomiony**. `create_model("D")` buduje tylko θ = 125. | `run_experiment.py:430-445` |
| K3 | K | Raport §5 proponuje „zakończenie badania na Etapie 1”, a §3 nazywa S1 „oficjalnym wnioskiem”. Tymczasem protokół §5.3 mówi, że przejście do Etapu 2 jest bezwarunkowe, a wnioski są wiążące dopiero przy N = 50. To optional stopping. | protokół §5.3; raport §3, §5 |
| K4 | K | H5 przetestowano połowicznie: tylko AE ≥ 2%. Warunek BCD < 0.1% nie jest testowany. Rzeczywistość: B 40.2%, C 6.3%, D 52.5% vs A 1.95%, E 1.68%. To odwrotność hipotezy i nie jest nazwane falsyfikacją. | `run_experiment.py` (p_h5); raport §2 |
| K5 | P | Raport pomija kontrasty atomowe sprzeczne z narracją: H2a (B − E = −0.48 p.p.: bank gorszy od ReLU) i H1a (luka BP − B = 2.74 p.p.). | `wyniki_kontrola_wymiaru_report.json` vs raport |
| K6 | P | p rodziny H3/H4 = `min(p_dir, p_TOST)`. Wybór korzystniejszego z dwóch testów bez korekty. | `run_experiment.py` (family_contrasts) |
| K7 | P | „Idealna dekompozycja addytywna” (D−B)+(C−D) = C−B to tożsamość algebraiczna, przedstawiona jako odkrycie. | raport §2B |
| K8 | P | Kontrola BP ma inną topologię (MLP 2×250) niż C (6 podwarstw). Margines H1 (2 p.p.) jest luźny: nawet A ma lukę 2.09 p.p. „Dorównuje backpropagacji” to nadinterpretacja. | `model_backprop_baseline.py`; raport §3 |
| K9 | P | Brak zbioru walidacyjnego. Wszystkie decyzje i ewaluacje na zbiorze testowym. Hiperparametry (θ, lr, wd) wybrane analitycznie lub arbitralnie, bez równego budżetu strojenia. | runner |
| K10 | P | Brak ramion rozdzielających „polimorfizm” od „kolejności” (np. odwrócona kaskada, bank w każdej podwarstwie, monokultury tanh/leaky w głębokim bloku). Kontrast C−D nie identyfikuje mechanizmu. | lista modeli |
| K11 | P | Warunkowy wybór testu (Shapiro → t albo Wilcoxon) zależny od danych. Rozkład p po takim dwuetapowym wyborze nie jest nominalny. | `paired_test` |
| K12 | P | Interpretacje sprzeczne z wynikami testów: IPR „dowodzi maksymalnej delokalizacji” przy H6 z p = 1.0; „zaledwie 6.26% martwych” przy wartości wyższej niż w obu modelach ReLU; „deklasacja Hintona”, gdy A to własna reimplementacja (95.8%), a Hinton (2022) raportuje 1.36% błędu testowego dla FF z etykietą w obrazie (4 × 2000 ReLU, 60 epok, dobroć sumowana bez pierwszej warstwy). | raport §4 |
| K13 | D | Ziarna losują tylko inicjalizację i kolejność batchy, nie podział danych. Wnioskowanie dotyczy wyłącznie zmienności treningu. | `get_dataloaders` |
| K14 | D | Definicja martwego neuronu zmieniona między eksperymentami (dodano var < 1e-4), więc wyniki nie są porównywalne z SCA. Progi absolutne nie są niezmiennicze względem skali aktywacji. | `compute_dead_neurons` |
| K15 | D | Czas mierzony na współdzielonej maszynie (`taskset`, `nice`, inne aplikacje). Teza „4× szybciej” jest tylko opisowa. | raport §4.2 |
| K16 | D | Progi H6 (0.005 w jednostkach IPR) i H2 (1.5 p.p.) bez uzasadnienia (brak SESOI). | protokół §4 |
| K17 | D | Protokół (§5.2) wymaga raportowania krzywych zbieżności co epokę. Raport ich nie pokazuje. | raport |
| K18 | D | Pula ziaren zawiera ziarna z pilota (42, 1337, 2024). | protokół §5.3 |

## D. Błędy procesowe wspólne dla całej serii

| ID | Waga | Błąd |
|---|---|---|
| G1 | P | „Zamrożenie” protokołu = lokalny commit git. Brak zewnętrznego znacznika czasu (OSF / publiczne repozytorium). Nie da się wykazać, że protokół nie był zmieniany. |
| G2 | P | Hipotezy kolejnych eksperymentów formułowane po obejrzeniu wyników poprzednich (co samo w sobie jest dopuszczalne), ale te same dane pilotowe są potem przedstawiane jako „potwierdzające”. |
| G3 | P | Brak szablonu raportu napisanego przed wynikami (results-blind). Raport pisany po wynikach, z narracją dopasowaną do wyników. |
| G4 | P | Brak testów kodu przed biegiem: liczba parametrów, brak złożeń liniowych, poprawność negatywów, analiza na danych symulowanych ze znanym efektem. |
| G5 | D | Rejestr językowy („absolutnie bezprecedensowa”, „dowodzi”, „idealna”) nieadekwatny do badania na MNIST z MLP. |
| G6 | D | Długie bibliografie bez weryfikacji, część pozycji nie jest powiązana z tezami. |

---

## Co z poprzednich danych da się uczciwie użyć

- **Jako dane pilotowe do analizy mocy:** SD różnic parowanych z Etapu 1 (N = 10) wynoszą 0.15–0.40 p.p. (np. C−B 0.30, C−D 0.34, B−E 0.28).
- **Jako obserwacje eksploracyjne (do zweryfikowania, nie do cytowania jako wynik):**
  - E (ReLU d=250) ≈ A (ReLU D=1000): różnica 0.16 p.p.
  - B (bank) < E o 0.48 p.p.
  - Blokowe kaskady (C, D) > płaskie modele FF.
  - Gładkie i nasycające aktywacje mają więcej (nie mniej) jednostek „martwych” wg definicji K14.
- **Czasy pojedynczych biegów (MNIST, 15 epok/makrowarstwę, 3 rdzenie i5-3450):** BP 95 s, E 186 s, B 216 s, C 242 s, D 243 s, A 958 s.
