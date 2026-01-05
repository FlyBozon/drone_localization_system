#!/usr/bin/env python3
"""
Demo script showing the complete workflow
This demonstrates how all components work together
"""

import os
import sys
import subprocess
from datetime import datetime

def print_section(title):
    print(title)

def run_command(cmd, description):
    print(f"\n{description}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n ERROR: {description} failed with code {result.returncode}")
        return False
    
    print(f"\n {description} completed successfully")
    return True

def main():
    print_section("MAP REFERENCE CREATOR - DEMO WORKFLOW")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis demo will process a small area in Gdańsk, Poland")
    print("Expected total time: ~3-6 minutes (with GPU) or ~16-31 minutes (CPU only)")
    
    # Parameters
    center_lat = 54.352
    center_lon = 18.646
    size_deg = 0.003
    zoom = 18
    osm_buffer = 0.15
    
    overlap = 64
    batch_size = 32
    strategy = "thesis"
    min_area = 50
    
    print(f"\nParameters:")
    print(f"  Location: {center_lat}, {center_lon}")
    print(f"  Size: {size_deg}° (~{size_deg * 111:.1f}km)")
    print(f"  Zoom: {zoom}")
    print(f"  Segmentation: overlap={overlap}, batch={batch_size}")
    print(f"  Merge: strategy={strategy}, min_area={min_area}")
    
    input("\nPress ENTER to start processing...")
    
    # Step 1: Download data
    print_section("STEP 1/3: Download orthophoto and OSM data")
    
    if not run_command(
        [sys.executable, "test_cli.py", 
         str(center_lat), str(center_lon), str(size_deg), str(zoom), str(osm_buffer)],
        "Downloading orthophoto and OSM data"
    ):
        return False
    
    # Find created folder
    coord_folders = [d for d in os.listdir('.')
                    if os.path.isdir(d) and d.startswith('lat_')]
    if not coord_folders:
        print("\n ERROR: No coordinate folder created")
        return False
    
    coord_folder = sorted(coord_folders)[-1]
    print(f"\nData saved to: {coord_folder}/")
    
    input("\nPress ENTER to continue to segmentation...")
    
    # Step 2: Segmentation
    print_section("STEP 2/3: Neural network segmentation")
    
    image_path = os.path.join(coord_folder, "image_orto.jpg")
    
    if not os.path.exists(image_path):
        print(f"\n ERROR: Orthophoto not found: {image_path}")
        return False
    
    if not run_command(
        [sys.executable, "segmentation_cli.py",
         image_path, coord_folder, str(overlap), str(batch_size)],
        "Running neural network segmentation (this may take a few minutes)"
    ):
        return False
    
    input("\nPress ENTER to continue to map merging...")
    
    # Step 3: Merge maps
    print_section("STEP 3/3: Merge topographic maps and NN segmentation")
    
    if not run_command(
        [sys.executable, "merge_topo_nn_cli.py",
         coord_folder, strategy, str(min_area)],
        "Merging maps"
    ):
        return False
    
    # Summary
    print_section("WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll results saved in: {coord_folder}/")
    print("\nGenerated files:")
    
    files_to_check = [
        ("image_orto.jpg", "Orthophoto"),
        ("segmentation_mask.png", "OSM topographic mask"),
        ("segmentation_nn_raw.png", "NN segmentation"),
        ("reference_map.png", " REFERENCE MAP (main result)"),
        ("reference_map_colored.png", "Reference map (colored)"),
        ("reference_map_comparison.png", "Comparison visualization"),
        ("metadata.txt", "Metadata"),
        ("reference_map_metadata.txt", "Reference map metadata")
    ]
    
    print()
    for filename, description in files_to_check:
        filepath = os.path.join(coord_folder, filename)
        status = "" if os.path.exists(filepath) else ""
        print(f"  {status} {filename:35} - {description}")
    
    print("You can now:")
    print(f"  1. Open {coord_folder}/reference_map_comparison.png to see all results")
    print(f"  2. Use {coord_folder}/reference_map.png as your reference map")
    print(f"  3. Check {coord_folder}/reference_map_metadata.txt for statistics")

    return True

if __name__ == "__main__":
    # Check if scripts exist
    required_scripts = ["test_cli.py", "segmentation_cli.py", "merge_topo_nn_cli.py"]
    missing = [s for s in required_scripts if not os.path.exists(s)]
    
    if missing:
        print("ERROR: Missing required scripts:")
        for script in missing:
            print(f"  - {script}")
        sys.exit(1)
    
    # Run demo
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
