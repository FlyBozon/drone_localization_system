# Instalacja Map Reference Creator

## Wymagania Systemowe

- Python 3.8 lub nowszy
- System operacyjny: Windows, Linux, lub macOS
- RAM: minimum 8GB (zalecane 16GB dla dużych obszarów)
- Dysk: ~5-10GB (model + dane tymczasowe)
- GPU (opcjonalnie): NVIDIA GPU z CUDA dla szybszej segmentacji

## Krok 1: Instalacja Pythona

### Windows:
1. Pobierz Python z https://www.python.org/downloads/
2. Podczas instalacji zaznacz "Add Python to PATH"
3. Zrestartuj komputer

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### macOS:
```bash
brew install python3
```

## Krok 2: Utworzenie Wirtualnego Środowiska (Zalecane)

```bash
# Utwórz wirtualne środowisko
python3 -m venv map_env

# Aktywuj środowisko
# Windows:
map_env\Scripts\activate
# Linux/macOS:
source map_env/bin/activate
```

## Krok 3: Instalacja Zależności

### Opcja A: Instalacja Podstawowa (bez GPU)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Opcja B: Instalacja z Obsługą GPU (NVIDIA)

**Uwaga**: Wymaga zainstalowanego CUDA Toolkit i cuDNN

```bash
pip install --upgrade pip

# Instalacja TensorFlow z GPU
pip install tensorflow-gpu>=2.10.0

# Pozostałe zależności
pip install PyQt5 PyQtWebEngine
pip install osmnx geopandas shapely
pip install opencv-python pillow numpy
pip install matplotlib scipy
pip install segmentation-models keras
pip install requests
```

### Weryfikacja Instalacji GPU:

```python
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## Krok 4: Instalacja Zależności Systemowych (Linux)

### Ubuntu/Debian:

```bash
# GDAL (wymagane przez geopandas/osmnx)
sudo apt install gdal-bin libgdal-dev

# Inne zależności
sudo apt install libspatialindex-dev python3-rtree
```

### Fedora/CentOS:

```bash
sudo dnf install gdal gdal-devel
sudo dnf install spatialindex spatialindex-devel
```

## Krok 5: Pobranie Modelu Sieci Neuronowej

Model jest wymagany do segmentacji. Możesz:

### Opcja A: Użyć własnego wytrenowanego modelu
Umieść model w:
```
trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras
```

albo zmień ścieżkę do modelu w kodzie.

### Opcja B: Trenowanie własnego modelu
Użyj skryptów treningowych z katalogu `nn_training/` 

### Opcja C: Model demo (do testów)
Utwórz plik `model.keras` w katalogu głównym

## Krok 6: Sprawdzenie Instalacji

```bash
# Test importów
python -c "import PyQt5; import osmnx; import cv2; import tensorflow; print('All imports successful!')"

# Test GUI (powinno otworzyć się okno z mapą)
python map_ref_creator_integrated.py
```

## Krok 7: Uruchomienie Demo

```bash
# Demo workflow (bez GUI)
python demo_workflow.py
```

## Rozwiązywanie Problemów

### Problem: "No module named 'PyQt5'"
```bash
pip install PyQt5 PyQtWebEngine
```

### Problem: "GDAL not found" (Linux)
```bash
# Ubuntu/Debian
sudo apt install gdal-bin libgdal-dev python3-gdal

# Lub zainstaluj przez pip (może wymagać kompilacji)
pip install gdal
```

### Problem: "ImportError: libGL.so.1"
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libglib2.0-0

# Fedora
sudo dnf install mesa-libGL
```

### Problem: TensorFlow nie widzi GPU
1. Sprawdź czy CUDA jest zainstalowany:
   ```bash
   nvcc --version
   ```

2. Sprawdź czy cuDNN jest zainstalowany:
   ```bash
   ldconfig -p | grep cudnn
   ```

3. Zainstaluj zgodne wersje:
   - TensorFlow 2.10: CUDA 11.2, cuDNN 8.1
   - TensorFlow 2.11+: CUDA 11.8, cuDNN 8.6

4. Sprawdź dokumentację TensorFlow:
   https://www.tensorflow.org/install/gpu

### Problem: OSMnx timeout errors
Zwiększ timeout w kodzie lub spróbuj ponownie - czasami serwery OSM są przeciążone.

### Problem: Out of memory podczas segmentacji
Zmniejsz `batch_size` w parametrach (np. z 32 do 16 lub 8)

## Struktura Katalogów

Po instalacji twoja struktura powinna wyglądać tak:

```
map_ref_creator/
├── map_ref_creator_integrated.py  # GUI
├── test_cli.py                    # Pobieranie danych
├── segmentation_cli.py            # Segmentacja NN
├── merge_topo_nn_cli.py          # Łączenie map
├── demo_workflow.py               # Demo
├── README.md                      # Dokumentacja
├── INSTALLATION.md               # Ten plik
├── requirements.txt              # Zależności
├── datasets_info.json           # Konfiguracja
└── trained_models/              # Folder na model
    └── landcover.ai_..._v1_early.keras
```

## Minimalne Wymagania Dyskowe

- Instalacja Pythona + biblioteki: ~3-5 GB
- Model sieci neuronowej: ~100-500 MB
- Dane tymczasowe (na obszar): ~50-200 MB
- **Łącznie**: ~5-7 GB wolnej przestrzeni

## Testowanie Instalacji

Uruchom pełny test:

```bash
# 1. Test importów
python -c "from PyQt5.QtWidgets import QApplication; from PyQt5.QtWebEngineWidgets import QWebEngineView; import osmnx; import tensorflow; import cv2; print(' All imports OK')"

# 2. Test GPU (jeśli zainstalowane)
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs: {len(gpus)}'); print(' GPU check OK' if len(gpus) > 0 else '⚠ No GPU, will use CPU')"

# 3. Test demo (bez GUI, używa małego obszaru)
python demo_workflow.py
```

## Następne Kroki

Po udanej instalacji:

1. Przeczytaj `README.md` dla instrukcji użytkowania
2. Uruchom `python map_ref_creator_integrated.py` dla GUI
3. LUB uruchom `python demo_workflow.py` dla demo CLI

## Wsparcie

W przypadku problemów:
1. Sprawdź logi błędów
2. Upewnij się że wszystkie zależności są zainstalowane
3. Sprawdź czy model sieci neuronowej jest dostępny
4. Dla problemów z GPU: sprawdź kompatybilność CUDA/cuDNN

## Aktualizacja

```bash
# Aktualizuj biblioteki
pip install --upgrade -r requirements.txt

# Lub pojedyncze pakiety
pip install --upgrade tensorflow osmnx PyQt5
```
