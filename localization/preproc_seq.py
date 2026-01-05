#preprocess images - crop to the same size
import os
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIG ---
input_dir = "output/dataset/moving_seq"
output_dir = "output/dataset/moving_seq/output_images"

border = 50                # px to remove from every side
target_width = 1000         # final crop width
target_height = 700        # final crop height

show_plots = True        

os.makedirs(output_dir, exist_ok=True)

exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(exts):
        continue

    in_path = os.path.join(input_dir, filename)
    out_path = os.path.join(output_dir, filename)

    img = Image.open(in_path)

    w, h = img.size
    img_no_border = img.crop((border, border, w - border, h - border))

    nb_w, nb_h = img_no_border.size

    if nb_w < target_width or nb_h < target_height:
        print(f"Skipping {filename}: image smaller than target size after border removal.")
        continue

    left = (nb_w - target_width) // 2
    top = (nb_h - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    cropped = img_no_border.crop((left, top, right, bottom))

    cropped.save(out_path)

    if show_plots:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Cropped")
        plt.imshow(cropped)
        plt.axis("off")

        plt.show()

print("Done.")
