# Szybki Start - Map Reference Creator


### Przygotowanie (Jednorazowe)

```bash
# 1. Zainstaluj zależności
pip install -r requirements.txt

# 2. Sprawdź czy działa
python -c "import PyQt5; import osmnx; import tensorflow; print(' OK')"
```

### Metoda 1: GUI (Najprostsza) 🖱️

```bash
python map_ref_creator_integrated.py
```

**4 kroki:**
1. Przesuń mapę do wybranego miejsca
2. Kliknij "Pobierz współrzędne z mapy"
3. Kliknij "Download current view"
4. Czekaj ~5 minut (GPU) lub ~20 minut (CPU)

**Gotowe!** Pliki znajdziesz w folderze `lat_XX_XXX_lon_YY_YYY/`

### Metoda 2: Demo Script (Automatyczne)

```bash
python demo_workflow.py
```

To uruchomi kompletny proces dla przykładowego obszaru w Gdańsku.

### Metoda 3: Linia Poleceń (Dla Zaawansowanych)

```bash
# Jedna komenda - pełny proces
python test_cli.py 54.352 18.646 0.003 18 && \
python segmentation_cli.py lat_*/image_orto.jpg lat_*/ && \
python merge_topo_nn_cli.py lat_*/ thesis 50
```

## Gdzie Są Moje Wyniki? 

Wszystko w folderze o nazwie `lat_XX_XXXXXX_lon_YY_YYYYYY/`:

```
lat_54_352000_lon_18_646000/
├── image_orto.jpg                    ← Ortofotomapa
├── segmentation_mask.png            ← Mapa OSM
├── segmentation_nn_raw.png          ← Segmentacja NN
├── reference_map.png                ← WYNIK!
├── reference_map_colored.png        ← Wynik kolorowy
└── reference_map_comparison.png     ← Porównanie
```

## Najważniejszy Plik: 

**`reference_map.png`** - To jest twoja mapa referencyjna!

## Co Oznaczają Kolory? 

- **Czarny** - Tło
- **Czerwony** - Budynki 
- **Zielony** - Roślinność/Lasy 
- **Niebieski** - Woda 
- **Szary** - Drogi 

## Przykładowe Lokalizacje 

### Gdańsk (domyślne)
```bash
python test_cli.py 54.352 18.646 0.003 18
```

### Warszawa
```bash
python test_cli.py 52.2297 21.0122 0.003 18
```

### Kraków
```bash
python test_cli.py 50.0619 19.9370 0.003 18
```

### Wrocław
```bash
python test_cli.py 51.1079 17.0385 0.003 18
```

## Najczęstsze Problemy i Rozwiązania

### "Model not found"
**Rozwiązanie**: Umieść model w `trained_models/` lub jako `model.keras`

### "No GPU detected"
**To OK!** System działa na CPU, tylko wolniej.

### "OSM timeout"
**Rozwiązanie**: Spróbuj ponownie (serwery OSM czasem są przeciążone)

### "Out of memory"
**Rozwiązanie**: 
```bash
# Zmniejsz batch_size
python segmentation_cli.py <image> <folder> 64 8  # zamiast 32
```

### GUI się zamyka
**To normalne!** Skrypty działają w tle. Patrz na terminal/konsolę.

## Parametry do Zabawy 🎮

### Rozmiar obszaru (`size_deg`)
- `0.001` = ~110m - mały
- `0.003` = ~330m - **zalecane**
- `0.005` = ~550m - duży
- `0.01` = ~1100m - bardzo duży (długie przetwarzanie!)

### Poziom zoomu
- `17` = mniej szczegółów, mniejsze pliki
- `18` = **zalecane**, dobry kompromis
- `19` = maksymalne szczegóły, duże pliki, długie przetwarzanie

### Strategia łączenia
- `thesis` = hierarchia zaufania, opisana w prace inżynierskiej
- `hybrid` = balans OSM i NN
- `osm_priority` = zawsze OSM jeśli dostępne
- `nn_priority` = zawsze NN jeśli dostępne

## Zmiana Strategii

```bash
# Po segmentacji możesz eksperymentować z różnymi strategiami:
python merge_topo_nn_cli.py lat_*/ thesis 50
python merge_topo_nn_cli.py lat_*/ hybrid 50
python merge_topo_nn_cli.py lat_*/ osm_priority 50
```

Każda utworzy nowy zestaw plików `reference_map*`.

## Pro Tips

1. **Zaczynaj od małych obszarów** (0.001-0.003°) żeby testować
2. **Używaj zoom 18** dla najlepszego balansu
3. **GPU = 5x szybciej** - warto jeśli dostępne
4. **Sprawdzaj `reference_map_comparison.png`** - pokazuje wszystkie etapy
5. **Batch_size 16** jeśli masz problemy z pamięcią
6. **OSM buffer 0.15** (15%) jest zwykle OK dla większości przypadków

## Następne Kroki

Teraz gdy masz działający system:

1. Przeczytaj `README.md` dla pełnej dokumentacji
2. Sprawdź `INSTALLATION.md` jeśli masz problemy
3. Eksperymentuj z różnymi lokalizacjami i parametrami
4. Używaj GUI dla wygody lub CLI dla automatyzacji