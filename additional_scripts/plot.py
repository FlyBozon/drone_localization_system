#!/usr/bin/env python3

import sys
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_IMAGES = [
    "lat_54_371503_lon_18_618262/segmentation_nn_raw.png",
    "lat_54_371503_lon_18_618262/segmentation_mask.png",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_IMAGES_PER_PAGE = 4


def load_image(path: Path):
    img = cv.imread(str(path), cv.IMREAD_UNCHANGED)

    if img is None:
        print(f"[WARN] Could not load image: {path}")
        return None

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img


def collect_image_paths(paths):
    image_paths = []

    for p in paths:
        if not p.exists():
            print(f"[WARN] Path does not exist: {p}")
            continue

        if p.is_dir():
            imgs = sorted(
                x for x in p.iterdir()
                if x.is_file() and x.suffix.lower() in IMAGE_EXTS
            )
            image_paths.extend(imgs)
        else:
            if p.suffix.lower() in IMAGE_EXTS:
                image_paths.append(p)

    return image_paths


def plot_images_in_batches(image_paths):
    if not image_paths:
        print("No valid images to plot.")
        return

    for i in range(0, len(image_paths), MAX_IMAGES_PER_PAGE):
        batch = image_paths[i : i + MAX_IMAGES_PER_PAGE]

        images = []
        titles = []

        for p in batch:
            img = load_image(p)
            if img is not None:
                images.append(img)
                titles.append(p.name)

        if not images:
            continue

        n = len(images)
        cols = 2
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(10, 5 * rows))

        for idx, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(rows, cols, idx + 1)

            if img.ndim == 2:
                plt.imshow(img, cmap="magma")
            else:
                plt.imshow(img)

            plt.title(title)
            plt.axis("off")

        plt.tight_layout()
        plt.show()   # ← każde 4 obrazy = osobne okno


def main():
    if len(sys.argv) > 1:
        input_paths = [Path(p) for p in sys.argv[1:]]
    else:
        input_paths = [Path(p) for p in DEFAULT_IMAGES]

    image_paths = collect_image_paths(input_paths)
    plot_images_in_batches(image_paths)


if __name__ == "__main__":
    main()
