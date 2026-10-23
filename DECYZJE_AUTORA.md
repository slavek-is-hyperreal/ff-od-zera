# Decyzje autora (do włączenia do kolejnej wersji protokołu)

## D1 – Sprzęt: Vast.ai (GPU), 23.10.2026

**Decyzja autora:** wszystkie etapy obliczeniowe (mini-pilot, A, B, C, E2) są wykonywane na wynajętym GPU w Vast.ai, a nie na lokalnym i5-3450.

**Limit finansowy (twardy):**
- saldo konta Vast.ai w dniu decyzji: 13.99 USD;
- dodatkowo najwyżej 100 PLN doładowania.

Przeliczenie: kurs średni NBP z połowy października 2026 to ok. 3.90 PLN/USD. Po uwzględnieniu spreadu walutowego karty (~2–4%) 100 PLN to ok. **24–25 USD**. Doładowanie w USD należy dobrać tak, żeby obciążenie karty w PLN nie przekroczyło 100 PLN.

**Łączny limit wydatków na całe badanie: 38 USD.** Wiążący jest wyłącznie limit łączny; podział poniżej jest planistyczny (zgodnie z §11.1 protokołu):

| Pozycja | Limit |
|---|---|
| Mini-pilot + pomiary sprzętu + testy powtarzalności (§9) | 5 USD |
| Etapy A + B | 12 USD |
| Etap C + E2 + E9 | 18 USD |
| Rezerwa (awarie, powtórki po błędzie kodu) | 3 USD |

**Reguły:**
1. Typ GPU wybierany przed mini-pilotem: najtańsza oferta spełniająca wymagania (≥ 8 GB VRAM, niezawodność ≥ 0.98, instancja on-demand, nie interruptible). Ten sam typ GPU i ten sam obraz Dockera (z hashem) przez cały etap (§8). Zmiana typu dopuszczalna tylko między etapami, z wpisem w `ODSTEPSTWA.md`.
2. Mini-pilot mierzy koszt na komplet ramion na wybranym GPU. Jeśli przewidywany koszt wszystkich etapów i analiz GPU (A, B, C, E2, E9) przy N z reguły §7.5 przekracza 33 USD (limit bez rezerwy), redukcje stosowane są przed zamrożeniem zgodnie z §11.3 protokołu. N nie jest redukowane.
3. Po zamrożeniu, przy przekroczeniu limitu w trakcie, badanie jest **wstrzymywane**, a nie skracane. Decyzja autora o ewentualnym dodatkowym budżecie trafia do `ODSTEPSTWA.md`.
4. Wyniki są synchronizowane lokalnie po każdym ukończonym biegu, z weryfikacją hashy (§8 protokołu). Instancja jest niszczona po zakończeniu każdego etapu, po commicie wyników.
5. Każde uruchomienie, doładowanie i zniszczenie instancji wymaga potwierdzenia autora.

**Wprowadzone w protokole (v4, uzupełnione w v5):** §5.7 (determinizm na GPU, `CUBLAS_WORKSPACE_CONFIG`), §8 (sprzęt = Vast.ai), §9.3 (mini-pilot na GPU; decyzja o sprzęcie już podjęta), §11 (budżet w USD zamiast godzin CPU).

## D2 – Kontrola „wyjścia tanh” w H1, 23.10.2026

**Decyzja autora:** do grupy komparatorów H1 dodane są ramiona `K-LLT` (LeakyReLU → LeakyReLU → tanh) i `K-GGT` (GELU → GELU → tanh).

**Cel:** H1 pyta wtedy, czy blok LGT daje więcej niż blok jednorodny poza ostatnią podwarstwą tanh. Odpowiedź odróżnia efekt kolejności i zestawu nieliniowości od samego efektu ograniczonej dobroci.

**Koszt:** +2 ramiona (16 → 18 ramion FF, ok. +12% obliczeń).

## D3 – Hipoteza „trasy” (efektywność parametrowa) jako analizy eksploracyjne, 23.10.2026

**Motywacja autora:** sieć jako nawigacja albo deformacja przestrzeni. ReLU daje tylko proste odcinki z ostrymi zakrętami: dojedzie wszędzie, ale nie najkrótszą drogą. Bogatszy zestaw nieliniowości powinien pozwolić na tę samą deformację mniejszą liczbą parametrów.

Analogia z korą mózgową była dopisana po fakcie i nie jest motywacją badania.

**Decyzja autora:** wariant A, czyli eksploracyjnie w tym badaniu:
- **E9:** przegląd szerokości na MNIST i Fashion-MNIST (GPU);
- **E10:** mały test dopasowania do spirali 2D, z rysunkiem map decyzyjnych do pracy (lokalnie, CPU).

Ewentualne osobne badanie konfirmacyjne (wariant B) zostaje odłożone do czasu wyników E9 i E10.
