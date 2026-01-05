Ten folder zawiera skrypty i narzędzia do lokalizacji drona na podstawie obrazów oraz analizy trajektorii. Wszystkie pliki są napisane w języku Python i służą do różnych etapów przetwarzania danych oraz oceny dokładności lokalizacji.

- **orb.py**  
  Główny skrypt do lokalizacji drona na podstawie detekcji cech ORB. Porównuje trajektorię wyznaczoną na podstawie obrazów z danymi ground truth, generuje wykresy i statystyki błędów. Obsługuje wariant z retrospektywną korektą oraz bez niej.

- **dopasowanie.py**  
  Skrypt do dopasowania fragmentów (patchy) obrazu do mapy referencyjnej na podstawie masek semantycznych. Zawiera kilka metod dopasowania (głosowanie, RANSAC, połączone), generuje wizualizacje oraz statystyki błędów lokalizacji.

- **preproc_seq.py**  
  Narzędzie do wstępnego przetwarzania sekwencji obrazów – przycina obrazy do zadanych rozmiarów i usuwa niepożądane obramowania. Opcjonalnie wyświetla podgląd przyciętych obrazów.

- **transform.py**  
  Skrypt do korekcji perspektywy dolnej części obrazu w selu późniejszego dopasowania go do referencji.