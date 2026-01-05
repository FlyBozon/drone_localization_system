# Map Reference Creator - Zintegrowany System

System do tworzenia map referencyjnych poprzez łączenie segmentacji sieci neuronowej z danymi topograficznymi OSM.

## Struktura Systemu

System składa się z 4 głównych komponentów:

1. **map_ref_creator_integrated.py** - GUI z mapą Leaflet do wyboru regionu
2. **test_cli.py** - Pobieranie ortofotomapy i danych OSM
3. **segmentation_cli.py** - Segmentacja ortofotomapy siecią neuronową
4. **merge_topo_nn_cli.py** - Łączenie map topograficznych z segmentacją NN

## Wymagania

### Biblioteki Python:
```bash
pip install PyQt5 PyQtWebEngine
pip install osmnx geopandas shapely
pip install opencv-python pillow numpy
pip install matplotlib scipy
pip install tensorflow segmentation-models
pip install requests
```

### Dodatkowe pliki:
- **Model sieci neuronowej**: `trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras`
- **Konfiguracja datasetu**: `datasets_info.json` (opcjonalnie)

## Sposób Użycia

### Metoda 1: Interfejs Graficzny (Zalecana)

```bash
python map_ref_creator_integrated.py
```

**Kroki:**
1. W lewym panelu zobaczysz mapę Leaflet
2. Przesuń i przybliż mapę do wybranego regionu
3. Kliknij "Pobierz współrzędne z mapy" - przeniesie to dane do formularza
4. (Opcjonalnie) Dostosuj parametry:
   - Szerokość/długość geograficzną
   - Rozmiar obszaru (w stopniach)
   - Poziom zoomu (14-19, zalecany: 18)
5. Kliknij "▶ ROZPOCZNIJ PRZETWARZANIE"
6. Obserwuj postęp w dzienniku operacji
7. Po zakończeniu otrzymasz potwierdzenie i informację o lokalizacji plików

**Uwaga**: GUI zamyka się po rozpoczęciu przetwarzania - to normalne! Kolejne skrypty działają w tle.

### Metoda 2: Linia Poleceń (Zaawansowana)

#### Krok 1: Pobieranie danych
```bash
python test_cli.py <center_lat> <center_lon> <size_deg> <zoom> [osm_buffer]
```

Przykład (Gdańsk):
```bash
python test_cli.py 54.352 18.646 0.003 18 0.15
```

#### Krok 2: Segmentacja
```bash
python segmentation_cli.py <image_path> <output_folder> [overlap] [batch_size]
```

Przykład:
```bash
python segmentation_cli.py lat_54_352000_lon_18_646000/image_orto.jpg lat_54_352000_lon_18_646000 64 32
```

#### Krok 3: Łączenie map
```bash
python merge_topo_nn_cli.py <coord_folder> [strategy] [min_area]
```

Przykład:
```bash
python merge_topo_nn_cli.py lat_54_352000_lon_18_646000 thesis 50
```

## Parametry

### test_cli.py
- `center_lat` - szerokość geograficzna środka (np. 54.352)
- `center_lon` - długość geograficzna środka (np. 18.646)
- `size_deg` - rozmiar obszaru w stopniach (np. 0.003 = ~330m x 330m)
- `zoom` - poziom zoomu 14-19 (18 zalecany dla szczegółów)
- `osm_buffer` - bufor OSM (domyślnie 0.15 = 15% rozszerzenia)

### segmentation_cli.py
- `image_path` - ścieżka do ortofotomapy
- `output_folder` - folder wyjściowy
- `overlap` - nakładanie się patchy (domyślnie 64)
- `batch_size` - rozmiar batcha (domyślnie 32)

### merge_topo_nn_cli.py
- `coord_folder` - folder ze współrzędnymi
- `strategy` - strategia łączenia:
  - `thesis` (domyślna) - hierarchia z pracy dyplomowej
  - `hybrid` - zbalansowane podejście
  - `osm_priority` - priorytet dla OSM
  - `nn_priority` - priorytet dla NN
  - `vote` - głosowanie
- `min_area` - minimalna powierzchnia obiektu w pikselach (domyślnie 50)

## Wyjściowe Pliki

System tworzy folder o nazwie `lat_XX_XXXXXX_lon_YY_YYYYYY` zawierający:

### Z test_cli.py:
- `image_orto.jpg` - ortofotomapa z Geoportalu
- `segmentation_mask.png` - maska topograficzna OSM (grayscale, klasy 0-4)
- `alignment_visualization.png` - wizualizacja dopasowania
- `metadata.txt` - metadane (współrzędne, rozdzielczość, itp.)

### Z segmentation_cli.py:
- `segmentation_nn_raw.png` - surowa segmentacja NN (grayscale)
- `segmentation_nn_colored.png` - kolorowa wizualizacja segmentacji
- `segmentation_nn_visualization.png` - porównanie oryginał vs segmentacja

### Z merge_topo_nn_cli.py:
- `reference_map.png` - **GŁÓWNY WYNIK** - mapa referencyjna (grayscale)
- `reference_map_colored.png` - kolorowa wizualizacja wyniku
- `reference_map_comparison.png` - porównanie wszystkich map
- `reference_map_metadata.txt` - statystyki i metadane wyniku

## Mapowanie Klas

```
0 - Tło/Unlabeled (czarny)
1 - Budynki (czerwony)
2 - Roślinność/Lasy (zielony)
3 - Woda (niebieski)
4 - Drogi (szary)
```

## Hierarchia Zaufania (Strategia "thesis")

1. **OSM - Woda** (najwyższe zaufanie - stabilne w czasie)
2. **OSM - Drogi** (średnie zaufanie - względnie stabilne)
3. **NN - Roślinność** (trudna do wyodrębnienia z OSM, widoczna na zdjęciach)
4. **OSM + NN - Budynki** (fuzja: tylko budynki potwierdzone przez OSM)
5. **NN - Tło** (wszystko inne)

## Rozwiązywanie Problemów

### "Model file not found"
Upewnij się, że model znajduje się w:
- `trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras`
- LUB `model.keras` w katalogu roboczym

### "No GPU detected"
System działa również na CPU, ale będzie wolniejszy (5-10x).
Segmentacja ~2000x2000px na GPU: ~2-5 minut, na CPU: ~15-30 minut.

### "OSM data fetch failed"
- Sprawdź połączenie internetowe
- Spróbuj zmniejszyć obszar (size_deg)
- Zwiększ osm_buffer jeśli dane na brzegach są niekompletne

### "Memory error"
- Zmniejsz batch_size w segmentation_cli.py (np. z 32 do 16 lub 8)
- Zmniejsz obszar pobierania (size_deg)

### GUI się zamyka
To normalne! Po kliknięciu "ROZPOCZNIJ PRZETWARZANIE", GUI przekazuje pracę do skryptów CLI,
które działają w tle. Obserwuj terminal/konsolę dla logów postępu.

## Informacje Techniczne

### Czas Przetwarzania (dla obszaru 0.003° ≈ 330m x 330m):
- Pobieranie ortofotomapy: ~10-30 sekund
- Pobieranie danych OSM: ~10-30 sekund
- Segmentacja NN (GPU): ~2-5 minut
- Segmentacja NN (CPU): ~15-30 minut
- Łączenie map: ~5-15 sekund

**Łącznie (GPU)**: ~3-6 minut
**Łącznie (CPU)**: ~16-31 minut

### Rozdzielczość:
- Zoom 18: ~1.2 m/piksel
- Zoom 19: ~0.6 m/piksel (większe pliki, dłuższe przetwarzanie)

### Zalecane rozmiary obszaru:
- 0.001° (~110m) - małe osiedle
- 0.003° (~330m) - dzielnica (zalecane)
- 0.005° (~550m) - duży obszar
- 0.01° (~1.1km) - bardzo duży obszar (długie przetwarzanie)

## Przykłady Lokalizacji

### Gdańsk:
```bash
python test_cli.py 54.352 18.646 0.003 18
```

### Warszawa:
```bash
python test_cli.py 52.2297 21.0122 0.003 18
```

### Kraków:
```bash
python test_cli.py 50.0619 19.9370 0.003 18
```

## Licencja

Dane ortofotomapowe: Geoportal (© GUGiK)
Dane OSM: OpenStreetMap contributors (ODbL)
