#script to convert deepglobe dataset to landcover.ai.v1 structure, so i can use same training code on it (with minor adjustments)
import os
import glob
import shutil

src_dir = "datasets/deepglobe"

images_dir = os.path.join(src_dir, "images")
masks_dir = os.path.join(src_dir, "masks")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

all_files = glob.glob(os.path.join(src_dir, "*.*"))

for file in all_files:
    filename = os.path.basename(file)
    
    if filename.endswith("_sat.jpg"):
        new_name = filename.replace("_sat", "")
        shutil.move(file, os.path.join(images_dir, new_name))
    
    elif filename.endswith("_mask.png"):
        new_name = filename.replace("_mask", "")
        shutil.move(file, os.path.join(masks_dir, new_name))