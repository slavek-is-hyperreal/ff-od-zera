# Heterogeniczność i kolejność nieliniowości w sieciach Forward-Forward

Badanie zaplanowane z góry (prerejestracja), autor: Sławomir Majcher.

*Preregistered study on heterogeneity and ordering of nonlinearities in Forward-Forward MLPs at a fixed parameter budget (MNIST, Fashion-MNIST). Documents are in Polish.*

## Status

| Etap | Stan |
|---|---|
| Audyt poprzednich eksperymentów | `00_AUDYT_BLEDOW.md` |
| Protokół | `PROTOKOL_v5.md` – kandydat do zamrożenia (5 rund recenzji: `recenzje/`, odpowiedzi: `odpowiedzi/`) |
| Decyzje autora | `DECYZJE_AUTORA.md` (D1–D4) |
| Kod i testy | `kod/` (testy §9.1: `kod/tests/`) |
| Mini-pilot 2.0 | w toku |
| **Zamrożenie protokołu** | tag `ff-od-zera-prereg` – zostanie utworzony po wpisaniu wartości z mini-pilota 2.0, **przed** Etapem A |

Historia tego repozytorium (daty pushów na GitHub) służy jako zewnętrzny znacznik czasu (PROTOKOL_v5 §10). Wiążąca jest wersja oznaczona tagiem `ff-od-zera-prereg`. Każda zmiana po tym tagu trafia do `ODSTEPSTWA.md`.

## Zawartość

- `00_AUDYT_BLEDOW.md` – błędy wcześniejszych, nieprerejestrowanych eksperymentów, które to badanie zastępuje
- `PROTOKOL_v1.md` … `PROTOKOL_v5.md` – kolejne wersje protokołu (obowiązuje v5)
- `RAPORT_SZABLON.md` – szablon raportu napisany przed wynikami
- `kod/` – implementacja (`ffexp/`), testy, skrypty wykonania
- `plany/` – plany biegów
- `dane/SHA256SUMS_DANE` – hashe plików danych (same dane: MNIST, Fashion-MNIST, QMNIST – z oficjalnych źródeł)
- `wyniki/` – wyniki pilotów (pilot techniczny, pilot spirali)

## Licencja

Kod: MIT. Dokumenty: CC BY 4.0.
