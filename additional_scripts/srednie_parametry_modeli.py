import os
import json
import numpy as np
from glob import glob

def summarize_models(input_dir, output_path="summary.json"):
    summary = []

    json_files = glob(os.path.join(input_dir, "*.json"))
    if not json_files:
        print(" Brak plików JSON w folderze.")
        return

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"\n MODEL: {data['model_name']}")

        model_summary = {
            "model_name": data.get("model_name"),
            "model_path": data.get("model_path"),
            "dataset": data.get("dataset"),
            "n_classes": data.get("n_classes"),
            "patch_size": data.get("patch_size"),
            "overlap": data.get("overlap"),
            "timestamp": data.get("timestamp"),

            # tu zapiszemy średnie metryki
            "mean_metrics": {},

            # tu zapiszemy średnią macierz konfuzji
            "mean_confusion_matrix": None
        }

        n_classes = data.get("n_classes")
        confusion_sum = np.zeros((n_classes, n_classes), dtype=np.float64)
        confusion_count = 0

        metrics_collect = {}
        image_results = data.get("image_results", [])

        for image in image_results:
            metrics = image.get("metrics", {})

            # zbieranie zwykłych metryk liczbowych
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_collect.setdefault(key, []).append(value)

            # zbieranie macierzy konfuzji
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                cm = np.array(cm, dtype=np.float64)
                confusion_sum += cm
                confusion_count += 1

        # liczenie średnich metryk
        for key, values in metrics_collect.items():
            if values:
                model_summary["mean_metrics"][key] = float(sum(values) / len(values))

        # średnia macierz konfuzji
        if confusion_count > 0:
            mean_cm = confusion_sum / confusion_count
            model_summary["mean_confusion_matrix"] = mean_cm.tolist()

            print("\n ŚREDNIA MACIERZ KONFUZJI:")
            print_matrix(mean_cm)

        # wypisywanie zbiorczych metryk
        print("\n ŚREDNIE METRYKI:")
        for m, v in model_summary["mean_metrics"].items():
            print(f"  {m:<20}: {v:.4f}")

        summary.append(model_summary)

    # zapis JSON
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n Zapisano zbiorcze podsumowanie do: {output_path}")


def print_matrix(mat):
    """ Ładne drukowanie macierzy konfuzji """
    mat = np.array(mat)
    rows, cols = mat.shape

    # szerokość formatowania
    cell_width = max(10, max(len(f"{v:.0f}") for v in mat.flatten()) + 2)

    header = " " * 8 + " ".join([f"PRED {j}".center(cell_width) for j in range(cols)])
    print(header)
    print("-" * len(header))

    for i in range(rows):
        row_label = f"TRUE {i}".ljust(7)
        row_values = " ".join([f"{mat[i, j]:>{cell_width}.0f}" for j in range(cols)])
        print(f"{row_label} {row_values}")


# PRZYKŁADOWE URUCHOMIENIE
if __name__ == "__main__":
    summarize_models(
        input_dir="additional_scripts/jsons",
        output_path="summary.json"
    )
