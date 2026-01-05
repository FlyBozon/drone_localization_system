# System lokalizacji agenta oparty na algorytmach wizji komputerowej

Repozytorium zawiera kod źródłowy do pracy inżynierskiej poświęconej opracowaniu systemu lokalizacji UAV wykorzystującego algorytmy wizji komputerowej, segmentację semantyczną oraz zdjęcia satelitarne i mapy topograficzne.

## Opis projektu

Celem projektu było opracowanie systemu lokalizacji wizyjnej dla bezzałogowych statków powietrznych (UAV) bazującego na dopasowywaniu zdjęć dronowych do map referencyjnych. System wykorzystuje:

- Segmentację semantyczną obrazów przy użyciu głębokich sieci neuronowych
- Algorytmy dopasowywania wizyjnego (ORB Visual Odometry)
- Korekcję perspektywy z widoku skośnego drona na widok nadirowy
- Fuzję danych z ortofotomap satelitarnych i map topograficznych OpenStreetMap

notatka: modele wytrenowane w trakcie pracy inżynierskiej do segmentacji semantycznej są dostępne pod tym linkiem, należy je pobrać oraz umieścić w folderze trained_models: https://drive.google.com/drive/folders/1zwQrbdYsJ6gKzKrEEchmzkGY7AS8qNdM?usp=drive_link

## Struktura projektu

### `nn_training/`
Moduł do trenowania sieci neuronowych do segmentacji semantycznej. Zawiera:
- `main.py` - główny skrypt treningowy
- `processor.py` - klasa DatasetProcessor do przetwarzania zbiorów danych (landcover.ai, UAVid, itp.)
- Obsługa wielu backbonów (ResNet, EfficientNet, MobileNet)
- Integracja z ClearML do monitorowania eksperymentów
- Wsparcie dla treningu multi-GPU (MirroredStrategy)

Obsługiwane zbiory danych:
- landcover.ai - segmentacja ortofotomap polskich
- UAVid - segmentacja zdjęć dronowych
- Deepglobe, Inria - dodatkowe zbiory do eksperymentów

### `lokalization_estimation/`
Algorytmy lokalizacji względnej i estymacji pozycji:
- `orb.py`- implementacja ORB Visual Odometry z retrospektywną korekcją
- `estimator.py` - estymator pozycji
- `dopasowanie.py` - algorytmy dopasowywania semantycznego (voting accumulator, RANSAC)
- `transform.py` - korekcja perspektywy z widoku skośnego na nadirowy
- `plot_trajectory.py`, `plot_orb_trajectory.py` - wizualizacja trajektorii

Moduł implementuje:
- Detekcję i dopasowywanie punktów charakterystycznych ORB
- Retrospektywną korekcję trajektorii
- Kalibrację kierunku ruchu na podstawie początkowych klatek
- Porównanie z danymi ground truth

### `gui/`
Interfejs graficzny do tworzenia map referencyjnych:
- `main.py` - główna aplikacja GUI (PyQt5)
- `map.py` - moduł pobierania map satelitarnych
- `segmentation.py` - segmentacja semantyczna w GUI
- `merge_topo_nn.py` - fuzja map topograficznych z wynikami segmentacji

Podkatalog `ref_map_creator/`:
- `demo_workflow.py` - skrypt demonstracyjny pełnego workflow
- `map_ref_creator_enhanced.py` - rozszerzony kreator map
- `map_ref_creator_integrated.py` - zintegrowana wersja
- Wersje CLI poszczególnych modułów (`*_cli.py`)

### `pipeline/`
Pipeline do pełnej analizy obrazów:
- `pipeline.sh` - skrypt bash uruchamiający kompletny proces
- `patch_matcher.py` - segmentacja i dopasowywanie patchy obrazu

Workflow pipeline:
1. Dzielenie obrazu na nakładające się patche
2. Segmentacja semantyczna każdego patcha
3. Dopasowywanie do mapy referencyjnej (voting, RANSAC)
4. Agregacja wyników

### `preprocess_imagery/`
Skrypty do preprocessing obrazów:
- `main.py` - główny moduł equalizacji histogramu (CLAHE)
- `drone.py`, `effects.py` - efekty i transformacje
- `preproc_for_uavid_train.py` - przygotowanie danych UAVid
- `preprocess_google_earth_imagery.py` - preprocessing obrazów satelitarnych

### `additional_scripts/`
Narzędzia pomocnicze:
- `plot.py` - generowanie wykresów do pracy
- `confusion_matrix.py` - macierze pomyłek
- `json_results_plot.py` - wizualizacja wyników treningów
- `get_param_nr_in_model.py` - analiza liczby parametrów modeli
- `srednie_parametry_modeli.py` - statystyki modeli

### `datasets/`
Zbiory danych:
- `landcover.ai.v1/` - ortofotomapy polskie
- `UAVid/` - sekwencje wideo z dronów
- `uav-visloc/` - zbiór do lokalizacji wizyjnej
- `datasets_info.json` - konfiguracja zbiorów danych

### `trained_models/`
Wytrenowane modele sieci neuronowych w formacie `.keras`

### `dat_processing/`
Przetwarzanie danych z czujników:
- `calibration/` - kalibracja kamer
- `distortion/` - korekcja dystorsji

## Wymagania

```
tensorflow>=2.16.0
keras>=3.0.0
opencv-python>=4.8.0
segmentation-models (instalacja z GitHub)
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
Pillow>=9.0.0
PyQt5 (dla GUI)
osmnx (dla pobrania map OSM)
clearml>=1.11.0
```

Pełna lista w pliku `requirements.txt`.

## Instalacja

```bash
# Klonowanie repozytorium
git clone <url>
cd engineering_project

# Utworzenie środowiska wirtualnego
python3 -m venv venv
source venv/bin/activate

# Instalacja zależności
pip install -r requirements.txt

# Instalacja segmentation-models z GitHub
pip install git+https://github.com/qubvel/segmentation_models
```

## Przykłady użycia

### Trenowanie modelu segmentacji

#### 1. Trening podstawowy (main.py)

```bash
cd nn_training

# Podstawowe użycie - UAVid dataset
python main.py \
  --dataset uavid \
  --config datasets_info.json \
  --architecture fpn \
  --backbone efficientnetb3 \
  --epochs 100 \
  --batch-size 32

# Z preprocessingiem danych
python main.py \
  --dataset landcover.ai \
  --config datasets_info.json \
  --architecture unet \
  --backbone resnet50 \
  --epochs 50 \
  --batch-size 16 \
  --preprocess-tiles \
  --tile-overlap 32 \
  --choose-useful \
  --split-data

# Transfer learning z pretrenowanym modelem
python main.py \
  --dataset uavid \
  --config datasets_info.json \
  --pretrained-model trained_models/landcover.ai_model.keras \
  --backbone efficientnetb0 \
  --epochs 30 \
  --batch-size 32 \
  --no-clearml
```

**Parametry:**
- `--dataset` - nazwa zbioru danych (uavid, landcover.ai, inria, deepglobe)
- `--config` - ścieżka do datasets_info.json
- `--architecture` - architektura modelu (unet, fpn, linknet, pspnet, deeplabv3, deeplabv3plus)
- `--backbone` - backbone sieci (efficientnetb3, resnet50, mobilenetv2, etc.)
- `--epochs` - liczba epok treningu
- `--batch-size` - rozmiar batcha
- `--patch-size` - rozmiar patcha (domyślnie 256)
- `--preprocess-tiles` - tworzenie tile'ów z obrazów
- `--tile-overlap` - overlap między tile'ami
- `--choose-useful` - filtrowanie użytecznych patchy
- `--split-data` - podział na train/val/test
- `--pretrained-model` - ścieżka do pretrenowanego modelu
- `--no-clearml` - wyłączenie integracji ClearML

#### 2. Fine-tuning INRIA (main_inria_fine_tuning.py)

```bash
# Fine-tuning na INRIA dataset
python main_inria_fine_tuning.py \
  --pretrained-model /path/to/pretrained_model.keras \
  --config datasets_info.json \
  --dataset inria \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --freeze-ratio 0.5 \
  --backbone efficientnetb0
```

**Parametry:**
- `--pretrained-model` - model do fine-tuningu (wymagane)
- `--config` - ścieżka do konfiguracji (wymagane)
- `--learning-rate` - learning rate (domyślnie 5e-5)
- `--freeze-ratio` - jaka część encodera ma być zamrożona (0.0-1.0)
- `--building-idx` - indeks klasy budynków (domyślnie 1)
- `--phase` - numer fazy treningu (do trackingu)

#### 3. Inference na obrazach (segment.py)

```bash
# Podstawowa segmentacja obrazów
python segment.py \
  --model-path trained_models/uavid_64_epochs_efficientnetb3_backbone_batch32_v1.keras \
  --images-dir datasets/test_images \
  --output-dir output/results

# Z custom parametrami
python segment.py \
  --model-path trained_models/my_model.keras \
  --images-dir datasets/images \
  --output-dir output/segmentation \
  --config datasets_info.json \
  --dataset-name uavid \
  --backbone efficientnetb3 \
  --patch-size 512 \
  --overlap 64 \
  --num-samples 5
```

**Parametry:**
- `--model-path` - ścieżka do modelu (wymagane)
- `--images-dir` - katalog z obrazami (wymagane)
- `--output-dir` - katalog wyjściowy (wymagane)
- `--config` - ścieżka do datasets_info.json
- `--dataset-name` - nazwa datasetu
- `--backbone` - backbone modelu
- `--patch-size` - rozmiar patcha (domyślnie 512)
- `--overlap` - overlap między patchami (domyślnie 64)
- `--num-samples` - liczba próbek do wizualizacji (domyślnie 3)

#### 4. Testowanie modeli (models_testing.py)

```bash
# Testowanie modeli na obrazach TIFF
python models_testing.py \
  --models-dir trained_models \
  --test-images-dir datasets/landcover.ai.v1/test/images \
  --test-masks-dir datasets/landcover.ai.v1/test/masks \
  --output-dir testing_results \
  --dataset-name landcover.ai \
  --patch-size 256 \
  --overlap 64 \
  --batch-size 32
```

**Parametry:**
- `--models-dir` - katalog z modelami (wymagane)
- `--test-images-dir` - katalog z testowymi obrazami (wymagane)
- `--test-masks-dir` - katalog z maskami ground truth (opcjonalne)
- `--output-dir` - katalog wyjściowy (domyślnie: testing_models_results_tiff)
- `--dataset-name` - filtr nazwy datasetu dla modeli (domyślnie: landcover.ai)
- `--n-classes` - liczba klas segmentacji (domyślnie: 5)
- `--patch-size` - rozmiar patcha (domyślnie: 256)
- `--overlap` - overlap między patchami (domyślnie: 64)
- `--batch-size` - rozmiar batcha (domyślnie: 32)
- `--max-patches` - max liczba zapisywanych patchy na obraz (domyślnie: 50)

#### 5. Testowanie UAVid (segm_uavid.py)

```bash
# Testowanie wielu modeli na UAVid
python segm_uavid.py \
  --models-dir /path/to/models \
  --images-dir /path/to/uavid/test/Images \
  --output-dir output/uavid_results \
  --masks-dir /path/to/uavid/test/Labels \
  --config datasets_info.json \
  --dataset-filter uavid \
  --num-samples 3
```

**Parametry:**
- `--models-dir` - katalog z modelami (wymagane)
- `--images-dir` - katalog z obrazami testowymi (wymagane)
- `--output-dir` - bazowy katalog wyjściowy (wymagane)
- `--masks-dir` - katalog z maskami (opcjonalne)
- `--config` - ścieżka do datasets_info.json (domyślnie: datasets_info.json)
- `--dataset-filter` - filtr modeli po nazwie datasetu (domyślnie: uavid)
- `--num-samples` - liczba próbek do wizualizacji na model (domyślnie: 3)

#### 6. Testowanie Landcover.ai (vn.py)

```bash
# Tryb pojedynczego obrazu
python vn.py \
  --model-path trained_models/landcover_model.keras \
  --config datasets_info.json \
  --mode single \
  --image-path datasets/landcover.ai.v1/images/M-33-7-A-d-2-3.tif \
  --mask-path datasets/landcover.ai.v1/masks/M-33-7-A-d-2-3.tif \
  --output-dir output/single_test \
  --overlap 64 \
  --batch-size 64

# Tryb folderu
python vn.py \
  --model-path trained_models/landcover_model.keras \
  --config datasets_info.json \
  --mode folder \
  --input-folder datasets/landcover.ai.v1/images \
  --mask-folder datasets/landcover.ai.v1/masks \
  --output-dir output/folder_test \
  --overlap 64 \
  --batch-size 32

# Tryb datasetu (UAV-VisLoc)
python vn.py \
  --model-path trained_models/model.keras \
  --config datasets_info.json \
  --mode dataset \
  --dataset-root datasets/uav-visloc \
  --output-dir output/dataset_test \
  --sequences 01,02,03 \
  --process-satellite \
  --process-drone

# Tryb batch
python vn.py \
  --model-path trained_models/model.keras \
  --config datasets_info.json \
  --mode batch \
  --image-list image1.tif image2.tif image3.tif \
  --mask-list mask1.tif mask2.tif mask3.tif \
  --output-dir output/batch_test
```

**Parametry:**
- `--model-path` - ścieżka do modelu (wymagane)
- `--config` - ścieżka do datasets_info.json (wymagane)
- `--mode` - tryb przetwarzania: single, folder, dataset, batch (wymagane)
- `--output-dir` - katalog wyjściowy (wymagane)
- `--overlap` - overlap między patchami (domyślnie: 64)
- `--batch-size` - rozmiar batcha (domyślnie: 32)

**Tryb single:**
- `--image-path` - ścieżka do obrazu
- `--mask-path` - ścieżka do maski (opcjonalne)

**Tryb folder:**
- `--input-folder` - folder z obrazami
- `--mask-folder` - folder z maskami (opcjonalne)

**Tryb dataset:**
- `--dataset-root` - root datasetu
- `--sequences` - sekwencje do przetworzenia ("all" lub lista oddzielona przecinkami)
- `--process-satellite` / `--no-satellite` - przetwarzanie zdjęć satelitarnych
- `--process-drone` / `--no-drone` - przetwarzanie zdjęć z drona

**Tryb batch:**
- `--image-list` - lista ścieżek do obrazów
- `--mask-list` - lista ścieżek do masek (opcjonalne)

### Tworzenie mapy referencyjnej (GUI)

```bash
cd gui
python main.py
```

### Tworzenie mapy referencyjnej (CLI)

```bash
cd gui/ref_map_creator
python demo_workflow.py
```

### Uruchomienie pipeline dopasowywania

```bash
./pipeline/pipeline.sh \
  --input datasets/uav-visloc/02/drone/02_0007.JPG \
  --output output/patches \
  --model trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras \
  --config nn_training/datasets_info.json \
  --reference datasets/uav-visloc/02/ref_map.png \
  --patch-size 256 \
  --overlap 64
```

### Lokalizacja ORB

```bash
cd lokalization_estimation
python orb.py
```

Parametry konfiguracyjne w pliku:
- `MAX_FRAMES` - liczba klatek do przetworzenia
- `N_INIT_FRAMES` - klatki do kalibracji początkowej
- `N_RETROSPECTIVE_CORRECTION` - częstotliwość korekcji retrospektywnej

## Struktura danych

### Format konfiguracji zbiorów (`datasets_info.json`)

```json
{
  "datasets": {
    "landcover.ai": {
      "name": "LandCover.ai",
      "classes": {
        "num_classes": 5,
        "class_names": ["background", "building", "woodland", "water", "road"]
      },
      "paths": {
        "dataset_dir": "datasets/landcover.ai.v1"
      },
      "data_format": {
        "image_format": "tif",
        "mask_format": "png"
      }
    }
  }
}
```

## Uwagi techniczne

- Projekt wykorzystuje ClearML do śledzenia eksperymentów treningowych
- Wsparcie dla GPU (CUDA) jest wymagane do efektywnego treningu
- Modele segmentacji używają architektury Unet z różnymi backbonami
- Pipeline dopasowywania implementuje kilka metod: voting accumulator, RANSAC translation, combined voting+RANSAC
- Korekcja perspektywy zakłada transformację homograficzną z dolnej części obrazu

## Autorstwo

Projekt inżynierski - Lokalizacja UAV oparta na wizji komputerowej i segmentacji semantycznej.

### AI
Kody były tworzone przy częściowym wsparciu ze strony Claude i ChatGPT, szczególnie dotyczy to kodow do szybkich testów, generowania wykresów, poprawy README i plików opisujących jak poprawnie uruchomic system. Wszystkie modyfikacje tworzone przez ai były sprawdzane w celu zapewnienia poprawnego działania.

## Licencja

Kod źródłowy dostępny wyłącznie w celach edukacyjnych i badawczych. 

Dostępne są też kody nie umieszczone w danym repo, tylko w wersji roboczej, dotyczące szczególnie testów każdego z podsystemów, optymalizacji sieci neuronowych, fine tuningu modeli, tworzenia symulatora lotu dronem w realistycznym środowisku na bazie Unreal Engine itd. Te prace wychodzą poza zakres danego projektu, dlatego nie zostaly uwzględnione w obecnej wersji. Dostęp możliwy po kontakcie osobistym.