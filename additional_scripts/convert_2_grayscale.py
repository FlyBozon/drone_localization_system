#script to convert deepglobe dataset to landcover.ai.v1 structure and convert RGB masks to indexed
import os
import glob
import shutil
import cv2
import numpy as np

data_dir = "/data/markryku/"
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - falling back to CPU")

data_dir = "/data/markryku/"

def convert_color_masks_gpu(masks_dir):
    print("Converting RGB masks to indexed grayscale (GPU-accelerated)...")
    
    # DeepGlobe exact color mapping
    colors = np.array([
        [0, 0, 0],        # unknown
        [0, 255, 255],    # urban
        [255, 255, 0],    # agriculture  
        [255, 0, 255],    # rangeland
        [0, 255, 0],      # forest
        [0, 0, 255],      # water
        [255, 255, 255]   # barren
    ], dtype=np.uint8)
    
    if GPU_AVAILABLE:
        colors_gpu = cp.asarray(colors)
    
    mask_files = glob.glob(os.path.join(masks_dir, "*.png"))
    converted_dir = f"{masks_dir}_indexed"
    os.makedirs(converted_dir, exist_ok=True)
    
    print(f"Found {len(mask_files)} mask files to convert")
    
    for i, mask_file in enumerate(mask_files):
        rgb_mask = cv2.imread(mask_file)
        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
        
        if GPU_AVAILABLE:
            # Transfer to GPU
            rgb_mask_gpu = cp.asarray(rgb_mask, dtype=cp.uint8)
            h, w = rgb_mask_gpu.shape[:2]
            grayscale_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
            
            # Vectorized comparison on GPU
            for class_id, color in enumerate(colors_gpu):
                # Broadcasting comparison - much faster than loops
                mask = cp.all(rgb_mask_gpu == color[cp.newaxis, cp.newaxis, :], axis=2)
                grayscale_mask_gpu[mask] = class_id
            
            # Transfer back to CPU
            grayscale_mask = cp.asnumpy(grayscale_mask_gpu)
            
        else:
            # CPU fallback
            h, w = rgb_mask.shape[:2]
            grayscale_mask = np.zeros((h, w), dtype=np.uint8)
            
            for class_id, color in enumerate(colors):
                mask = np.all(rgb_mask == color, axis=2)
                grayscale_mask[mask] = class_id
        
        output_path = os.path.join(converted_dir, os.path.basename(mask_file))
        cv2.imwrite(output_path, grayscale_mask)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processed {i + 1}/{len(mask_files)} masks")
    
    print(f"Converted masks saved to: {converted_dir}")
    return converted_dir

def convert_structure():
    print("Converting DeepGlobe structure...")
    
    src_dir = f"{data_dir}datasets/deepglobe/train/"
    images_dir = os.path.join(src_dir, "images")
    masks_dir = os.path.join(src_dir, "masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(src_dir, "*.*"))
    
    moved_images = 0
    moved_masks = 0

    for file in all_files:
        filename = os.path.basename(file)
        
        if filename.endswith("_sat.jpg"):
            new_name = filename.replace("_sat", "")
            shutil.move(file, os.path.join(images_dir, new_name))
            moved_images += 1
        
        elif filename.endswith("_mask.png"):
            new_name = filename.replace("_mask", "")
            shutil.move(file, os.path.join(masks_dir, new_name))
            moved_masks += 1
    
    print(f"Moved {moved_images} images and {moved_masks} masks")
    return masks_dir

def convert_color_masks(masks_dir):
    print("Converting RGB masks to indexed grayscale...")
    
    # DeepGlobe exact color mapping
    color_to_class = {
        (0, 0, 0): 0,        # unknown
        (0, 255, 255): 1,    # urban
        (255, 255, 0): 2,    # agriculture  
        (255, 0, 255): 3,    # rangeland
        (0, 255, 0): 4,      # forest
        (0, 0, 255): 5,      # water
        (255, 255, 255): 6   # barren
    }
    
    mask_files = glob.glob(os.path.join(masks_dir, "*.png"))
    converted_dir = f"{masks_dir}_indexed"
    os.makedirs(converted_dir, exist_ok=True)
    
    print(f"Found {len(mask_files)} mask files to convert")
    
    for i, mask_file in enumerate(mask_files):
        rgb_mask = cv2.imread(mask_file)
        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
        
        h, w = rgb_mask.shape[:2]
        grayscale_mask = np.zeros((h, w), dtype=np.uint8)
        
        for color, class_id in color_to_class.items():
            color_array = np.array(color)
            mask = np.all(rgb_mask == color_array, axis=2)
            grayscale_mask[mask] = class_id
        
        output_path = os.path.join(converted_dir, os.path.basename(mask_file))
        cv2.imwrite(output_path, grayscale_mask)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processed {i + 1}/{len(mask_files)} masks")
    
    print(f"Converted masks saved to: {converted_dir}")
    return converted_dir

def verify_conversion(converted_dir):
    print("Verifying conversion...")
    
    mask_files = glob.glob(os.path.join(converted_dir, "*.png"))
    sample_files = mask_files[:3]  #first 3 files
    
    for mask_file in sample_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_values = np.unique(mask)
            filename = os.path.basename(mask_file)
            print(f"  {filename}: classes found {unique_values}")
            
            if np.max(unique_values) > 6:
                print(f"    WARNING: Found class {np.max(unique_values)} but max should be 6")
        else:
            print(f"    ERROR: Could not read {os.path.basename(mask_file)}")

def main():
    print("Starting DeepGlobe dataset conversion...")
    masks_dir = convert_structure()
    converted_dir = convert_color_masks_gpu(masks_dir)
    
    verify_conversion(converted_dir)
    
    print("\nConversion complete!")
    print(f"Images: datasets/deepglobe/images/")
    print(f"Original masks: datasets/deepglobe/masks/")
    print(f"Indexed masks: {converted_dir}")
    print("\nUpdate your datasets_info.json to point to the indexed masks directory.")

if __name__ == "__main__":
    main()