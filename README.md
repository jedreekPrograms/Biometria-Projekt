# Porównanie metod wykrywania syntetycznej mowy

Miniprojekt z przedmiotu Systemy Identyfikacyjne i Biometryczne. Eksperyment na ASVspoof 2019 Logical Access porównuje pięć zestawów cech przy stałych podziałach i parametrach lasu losowego. Prototyp klasyfikuje plik WAV lub FLAC.

[Raport PDF](Raport-wykrywanie-mowy-syntetycznej.pdf) · [Kompletny pakiet ZIP](https://github.com/jedreekPrograms/Biometria-Projekt/releases/download/v1.0.0/Miniprojekt-ASVspoof-LA.zip)

## Uruchomienie na Windows

Rozpakuj archiwum i otwórz PowerShell w folderze z plikiem `detektor.py`. Użyj Pythona 3.12 w wersji 64-bitowej.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe detektor.py ocena "C:\sciezka\nagranie.wav"
.\.venv\Scripts\python.exe detektor.py ocena "C:\sciezka\nagranie.flac" --wariant A
```

Domyślnie ładowany jest wariant E. Program zwraca etykietę, wynik lasu dla klasy syntetycznej, próg z walidacji oraz liczbę segmentów mowy i pauz. Wynik lasu można porównać z progiem decyzji. Nagranie krótsze niż 2 s, sygnał praktycznie cichy lub brak wykrytej mowy skutkują wynikiem „brak oceny” i kodem wyjścia 2. Korzystaj z plików modeli dostarczonych w tym archiwum lub wytrenowanych lokalnie.

## Odtworzenie wyników

Trening można powtórzyć bez pobierania danych, używając dostarczonej tabeli cech. Do powtórzenia ekstrakcji pobierz [LA.zip z Edinburgh DataShare](https://doi.org/10.7488/ds/2555) (około 7,1 GiB, MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5`). Skrypt obsługuje archiwum i wypakowany katalog `LA`.

```powershell
.\.venv\Scripts\python.exe detektor.py ekstrakcja "C:\sciezka\LA.zip" --procesy 8
.\.venv\Scripts\python.exe detektor.py trening --procesy 8
.\.venv\Scripts\python.exe wykresy.py
.\.venv\Scripts\python.exe raport.py
```

Ekstrakcja trwa kilka minut lub dłużej zależnie od komputera. Ponowne uruchomienie domyślnie aktualizuje pliki w `wyniki` i `modele`. Polecenie `detektor.py <polecenie> --help` pokazuje opcje wyboru katalogów i liczby procesów. Czasy ekstrakcji różnią się między uruchomieniami. Na Linux i macOS odpowiednikiem ścieżki do środowiska jest `.venv/bin/python`.

## Pliki wynikowe

| Plik | Zawartość |
|---|---|
| `cechy.csv.gz` | Jeden wiersz na przyjęte nagranie. Cechy i metadane zapisano w osobnych kolumnach. |
| `kontrola_danych.csv.gz`, `liczebnosci.csv` | Czas nagrania, pauzy, ramki dźwięczne i status po filtracji lub przyczyna odrzucenia. |
| `metryki.csv` | Metryki i macierze pomyłek dla wariantów A–E, trzech ziaren i wszystkich scenariuszy. |
| `generatory.csv`, `bledy.csv` | Wyniki dla poszczególnych generatorów oraz według długości i obecności pauz. |
| `predykcje.csv.gz` | Wynik modelu, decyzja i próg dla każdego nagrania walidacyjnego i testowego. |
| `przedzialy.json` | Sparowane przedziały bootstrapowe dla różnicy balanced accuracy E–A. |
| `wykresy/` | Rozkłady długości, porównanie metryk, macierze i wyniki według generatora. |
| `modele/*.joblib` | Pięć wytrenowanych modeli i próg każdego wariantu dla ziarna 42. |

Klasa `0` oznacza mowę naturalną, a `1` syntetyczną lub konwertowaną. `eval_unseen` korzysta z próbek rzeczywistych i ataków A07–A15, A17–A18. `eval_reference` korzysta z tych samych próbek rzeczywistych oraz A16 i A19, które używają algorytmów znanych z A04 i A06. `eval_all` łączy oba zestawy ataków bez powielania nagrań naturalnych. EER obliczany jest z krzywej ROC przez interpolację, a pozostałe metryki używają progu wybranego na walidacji.

## Źródła

- Dane i protokoły są pochodną ASVspoof 2019 udostępnionego przez University of Edinburgh CSTR. Zobacz [rekord zbioru](https://doi.org/10.7488/ds/2555), [licencję ODC Attribution](https://opendatacommons.org/licenses/by/1-0/) i [opis algorytmów](https://arxiv.org/html/1911.01601). Pliki audio pobiera się osobno z repozytorium wydawcy.
- Definicje cech czasowych pochodzą z pracy *Investigation of Deepfake Voice Detection Using Speech Pause Patterns*. Zobacz [publikację](https://doi.org/10.2196/56245).
- `zasoby/silero_vad.onnx` to plik `silero_vad_16k_sequence.onnx` z [Silero VAD v6.2.2](https://github.com/snakers4/silero-vad/releases/tag/v6.2.2), rozpowszechniany na licencji MIT. SHA-256 `9ccdacc4719d8aa7e45a77536bfabec45a03ba1f2fad5e241ab4060b24238a85`.
