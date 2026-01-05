import os
import json
import re
import matplotlib.pyplot as plt

INPUT_FOLDER = "."
OUTPUT_FOLDER = "plots"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]

for filename in files:
    filepath = os.path.join(INPUT_FOLDER, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    name_no_ext = filename.replace(".json", "").lower()

    model_name = name_no_ext.split("_")[0]

    if "uavid" in name_no_ext:
        dataset = "uavid"
    elif "landcoverai" in name_no_ext:
        dataset = "landcoverai"
    elif "deepglobe" in name_no_ext:
        dataset = "deepglobe"
    else:
        dataset = "?"

    tile_size = None

    if "256" in name_no_ext:
        tile_size = "256"
    elif "512" in name_no_ext:
        tile_size = "512"
    else:
        if dataset == "uavid":
            tile_size = "512"
        elif dataset == "landcoverai":
            tile_size = "256"
        elif dataset == "deepglobe":
            tile_size = "256"
        else:
            tile_size = "?"

    if "iou" in name_no_ext:
        metric_title = "IoU w kolejnych epokach"
    elif "accuracy" in name_no_ext:
        metric_title = "Dokładność w kolejnych epokach"
    else:
        metric_title = "Wynik w kolejnych epokach"

    plot_title = f"{metric_title} – {model_name} | {dataset} | {tile_size}"

    plt.figure(figsize=(10, 6))

    for entry in data:
        x = entry["x"]
        y = entry["y"]
        label = entry.get("name", "Wynik")

        plt.plot(x, y, marker="o", linewidth=1, label=label)

    plt.suptitle(plot_title, fontsize=16, fontweight='bold')
    plt.xlabel("Epoka")
    plt.ylabel("Wartość")
    plt.grid(True)
    plt.legend(title="Metryki")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_name = filename + ".png"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Zapisano: {output_path}")

print("\Done (yuppy)")

