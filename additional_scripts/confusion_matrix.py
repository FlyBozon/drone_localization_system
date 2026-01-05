import os
import json
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def load_mean_confusion_matrices(input_dir):
    """Load all mean confusion matrices from JSON files."""
    json_files = glob(os.path.join(input_dir, "*.json"))

    models = []
    matrices = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        model_name = data["model_name"]

        # compute average confusion matrix manually (image_results)
        n_classes = data["n_classes"]
        cm_sum = np.zeros((n_classes, n_classes), dtype=np.float64)
        count = 0

        for image in data["image_results"]:
            cm = np.array(image["metrics"]["confusion_matrix"], dtype=np.float64)
            cm_sum += cm
            count += 1

        if count > 0:
            mean_cm = cm_sum / count
            models.append(model_name)
            matrices.append(mean_cm)

    return models, matrices


def plot_confusion_matrices(models, matrices, save_path="all_confusion_matrices.png"):
    """Plot multiple confusion matrices in one figure."""

    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i, (name, cm) in enumerate(zip(models, matrices)):
        ax = axes[i]
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # add numbers
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, f"{cm[r,c]:.0f}",
                        ha="center", va="center", fontsize=7, color="white")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f" Saved plot: {save_path}")


# RUNNING
if __name__ == "__main__":
    input_dir = "additional_scripts/jsons"

    models, matrices = load_mean_confusion_matrices(input_dir)
    plot_confusion_matrices(models, matrices)
