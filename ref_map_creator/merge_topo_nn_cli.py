#!/usr/bin/env python3


import numpy as np
import cv2
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


class ReferenceMapCreator:
    def __init__(self):
        self.CLASS_MAPPING = {
            0: 'unlabeled',
            1: 'buildings',
            2: 'woodlands',
            3: 'water',
            4: 'roads'
        }
        
        self.CLASS_COLORS_RGB = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (128, 128, 128)
        }
        
    def load_masks(self, coord_folder):
        nn_mask_path = os.path.join(coord_folder, "segmentation_nn_raw.png")
        osm_mask_path = os.path.join(coord_folder, "segmentation_mask.png")
        orto_path = os.path.join(coord_folder, "image_orto.jpg")
        
        if not os.path.exists(nn_mask_path):
            raise FileNotFoundError(f"NN mask not found: {nn_mask_path}")
        if not os.path.exists(osm_mask_path):
            raise FileNotFoundError(f"OSM mask not found: {osm_mask_path}")
        if not os.path.exists(orto_path):
            print(f"Warning: Ortophoto not found: {orto_path}")
            orto_image = None
        else:
            orto_image = cv2.imread(orto_path)
            orto_image = cv2.cvtColor(orto_image, cv2.COLOR_BGR2RGB)
        
        nn_mask = cv2.imread(nn_mask_path, cv2.IMREAD_GRAYSCALE)
        osm_mask = cv2.imread(osm_mask_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"Loaded NN mask: {nn_mask.shape}, unique values: {np.unique(nn_mask)}")
        print(f"Loaded OSM mask: {osm_mask.shape}, unique values: {np.unique(osm_mask)}")
        
        if nn_mask.shape != osm_mask.shape:
            print(f"Resizing masks to match: NN {nn_mask.shape} -> OSM {osm_mask.shape}")
            nn_mask = cv2.resize(nn_mask, (osm_mask.shape[1], osm_mask.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        return nn_mask, osm_mask, orto_image
    
    def create_reference_map(self, nn_mask, osm_mask, strategy='hybrid', min_area=50):
        if strategy == 'hybrid':
            reference_mask = self._hybrid_merge(nn_mask, osm_mask)
        elif strategy == 'thesis':
            reference_mask = self._thesis_fusion(nn_mask, osm_mask, min_area)
        elif strategy == 'osm_priority':
            reference_mask = osm_mask.copy()
            reference_mask[osm_mask == 0] = nn_mask[osm_mask == 0]
        elif strategy == 'nn_priority':
            reference_mask = nn_mask.copy()
            reference_mask[nn_mask == 0] = osm_mask[nn_mask == 0]
        elif strategy == 'vote':
            reference_mask = self._voting_merge(nn_mask, osm_mask)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return reference_mask
    
    def _thesis_fusion(self, nn_mask, osm_mask, min_area=50):
        reference_mask = nn_mask.copy()
        
        print("\n  Applying thesis fusion strategy:")
        print(f"    Step 1: Initialize with NN segmentation")
        
        # OSM class mapping: 1=water, 2=vegetation, 3=roads, 4=buildings
        # NN class mapping: 0=background, 1=buildings, 2=woodlands, 3=water, 4=roads
        
        # Step 2: Water from OSM
        osm_water = (osm_mask == 3)
        water_pixels_before = np.sum(reference_mask == 3)
        reference_mask[osm_water] = 3
        water_pixels_after = np.sum(reference_mask == 3)
        print(f"    Step 2: Water from OSM - added {water_pixels_after - water_pixels_before} pixels")
        
        # Step 3: Roads from OSM
        osm_roads = (osm_mask == 4)
        roads_pixels_before = np.sum(reference_mask == 4)
        reference_mask[osm_roads] = 4
        roads_pixels_after = np.sum(reference_mask == 4)
        print(f"    Step 3: Roads from OSM - added {roads_pixels_after - roads_pixels_before} pixels")
        
        # Step 4: Building correction - only OSM-confirmed buildings
        nn_buildings = (nn_mask == 1)
        osm_buildings = (osm_mask == 1)
        
        unconfirmed_buildings = nn_buildings & ~osm_buildings
        buildings_removed = np.sum(unconfirmed_buildings)
        
        if buildings_removed > 0:
            for y in range(reference_mask.shape[0]):
                for x in range(reference_mask.shape[1]):
                    if unconfirmed_buildings[y, x]:
                        osm_value = osm_mask[y, x]
                        if osm_value != 0:
                            reference_mask[y, x] = osm_value
                        else:
                            neighborhood = self._get_nearest_non_building_class(osm_mask, y, x)
                            if neighborhood != 0 and neighborhood != 1:
                                reference_mask[y, x] = neighborhood
        
        buildings_after = np.sum(reference_mask == 1)
        print(f"    Step 4: Building correction - removed {buildings_removed} unconfirmed buildings")
        print(f"              Remaining buildings: {buildings_after}")
        
        # Step 5: Filter small objects
        if min_area > 0:
            reference_mask = self._filter_small_objects(reference_mask, osm_mask, min_area)
            print(f"    Step 5: Small object filtering (min_area={min_area} pixels)")
        
        return reference_mask
    
    def _get_nearest_non_building_class(self, mask, y, x, max_dist=5):
        h, w = mask.shape
        
        for dist in range(1, max_dist + 1):
            y_min = max(0, y - dist)
            y_max = min(h, y + dist + 1)
            x_min = max(0, x - dist)
            x_max = min(w, x + dist + 1)
            
            neighborhood = mask[y_min:y_max, x_min:x_max]
            valid_classes = neighborhood[(neighborhood != 0) & (neighborhood != 1)]
            
            if len(valid_classes) > 0:
                unique, counts = np.unique(valid_classes, return_counts=True)
                return unique[np.argmax(counts)]
        
        return 0
    
    def _filter_small_objects(self, reference_mask, osm_mask, min_area=50):
        filtered_mask = reference_mask.copy()
        removed_objects = 0
        
        for class_id in [1, 2, 3, 4]:
            class_mask = (reference_mask == class_id).astype(np.uint8)
            osm_class_mask = (osm_mask == class_id).astype(np.uint8)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                
                if area < min_area:
                    component_mask = (labels == i)
                    osm_overlap = np.sum(osm_class_mask[component_mask])
                    
                    if osm_overlap == 0:
                        filtered_mask[component_mask] = 0
                        removed_objects += 1
        
        if removed_objects > 0:
            print(f"      Removed {removed_objects} small objects")
        
        return filtered_mask
    
    def _hybrid_merge(self, nn_mask, osm_mask):
        reference_mask = np.zeros_like(nn_mask)
        
        # Priority: OSM > NN for certain classes
        osm_priority_classes = [3, 4]  # Water, Roads
        
        for class_id in osm_priority_classes:
            osm_class = (osm_mask == class_id)
            reference_mask[osm_class] = class_id
        
        # NN for vegetation
        nn_vegetation = (nn_mask == 2) & (reference_mask == 0)
        reference_mask[nn_vegetation] = 2
        
        # Intersection for buildings
        nn_buildings = (nn_mask == 1)
        osm_buildings = (osm_mask == 1)
        buildings = nn_buildings & osm_buildings
        reference_mask[buildings] = 1
        
        # Fill remaining with NN
        remaining = (reference_mask == 0) & (nn_mask != 0)
        reference_mask[remaining] = nn_mask[remaining]
        
        return reference_mask
    
    def _voting_merge(self, nn_mask, osm_mask):
        reference_mask = np.zeros_like(nn_mask)
        
        agree = (nn_mask == osm_mask) & (nn_mask != 0)
        reference_mask[agree] = nn_mask[agree]
        
        disagree = (nn_mask != osm_mask) & (nn_mask != 0) & (osm_mask != 0)
        reference_mask[disagree] = osm_mask[disagree]
        
        return reference_mask
    
    def create_colored_mask(self, mask):
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in self.CLASS_COLORS_RGB.items():
            colored[mask == class_id] = color
        return colored
    
    def print_statistics(self, mask, title):
        print(f"\n{title}:")
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size
        
        for class_id, count in zip(unique, counts):
            class_name = self.CLASS_MAPPING.get(class_id, f'Unknown_{class_id}')
            percentage = (count / total) * 100
            print(f"  {class_name}: {count} pixels ({percentage:.2f}%)")
    
    def visualize_comparison(self, nn_mask, osm_mask, reference_mask, orto_image=None, save_path=None):
        if orto_image is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
        axes = axes.flatten()
        
        if orto_image is not None:
            axes[0].imshow(orto_image)
            axes[0].set_title('(a) Original Orthophoto', fontsize=14)
            axes[0].axis('off')
            
            nn_colored = self.create_colored_mask(nn_mask)
            axes[1].imshow(nn_colored)
            axes[1].set_title('(b) NN Segmentation', fontsize=14)
            axes[1].axis('off')
            
            osm_colored = self.create_colored_mask(osm_mask)
            axes[2].imshow(osm_colored)
            axes[2].set_title('(c) OSM Topographic Map', fontsize=14)
            axes[2].axis('off')
            
            ref_colored = self.create_colored_mask(reference_mask)
            axes[3].imshow(ref_colored)
            axes[3].set_title('(d) Reference Map (Merged)', fontsize=14)
            axes[3].axis('off')
            
            axes[4].imshow(orto_image)
            overlay = self.create_colored_mask(reference_mask)
            axes[4].imshow(overlay, alpha=0.5)
            axes[4].set_title('(e) Reference Map Overlay', fontsize=14)
            axes[4].axis('off')
            
            # Source visualization
            source_map = np.zeros((*reference_mask.shape, 3), dtype=np.uint8)
            
            nn_used = (reference_mask != 0) & (reference_mask == nn_mask) & (reference_mask != osm_mask)
            source_map[nn_used] = [0, 255, 0]  # Green - NN
            
            osm_used = (reference_mask != 0) & (reference_mask == osm_mask) & (reference_mask != nn_mask)
            source_map[osm_used] = [0, 0, 255]  # Blue - OSM
            
            both_used = (reference_mask != 0) & (reference_mask == nn_mask) & (reference_mask == osm_mask)
            source_map[both_used] = [128, 0, 128]  # Purple - Both agree
            
            fusion_used = (reference_mask != 0) & (reference_mask != nn_mask) & (reference_mask != osm_mask)
            source_map[fusion_used] = [255, 255, 0]  # Yellow - Fusion
            
            background = (reference_mask == 0)
            source_map[background] = [50, 50, 50]  # Dark gray
            
            axes[5].imshow(source_map)
            total = reference_mask.size
            osm_percent = (np.sum(osm_used) / total) * 100
            nn_percent = (np.sum(nn_used) / total) * 100
            axes[5].set_title(f'(f) Data Source\nOSM: {osm_percent:.1f}%, NN: {nn_percent:.1f}%', 
                             fontsize=12)
            axes[5].axis('off')
        else:
            nn_colored = self.create_colored_mask(nn_mask)
            axes[0].imshow(nn_colored)
            axes[0].set_title('(a) NN Segmentation', fontsize=14)
            axes[0].axis('off')
            
            osm_colored = self.create_colored_mask(osm_mask)
            axes[1].imshow(osm_colored)
            axes[1].set_title('(b) OSM Topographic Map', fontsize=14)
            axes[1].axis('off')
            
            ref_colored = self.create_colored_mask(reference_mask)
            axes[2].imshow(ref_colored)
            axes[2].set_title('(c) Reference Map (Merged)', fontsize=14)
            axes[2].axis('off')
            
            # Source visualization
            source_map = np.zeros((*reference_mask.shape, 3), dtype=np.uint8)
            
            nn_used = (reference_mask != 0) & (reference_mask == nn_mask) & (reference_mask != osm_mask)
            source_map[nn_used] = [0, 255, 0]
            
            osm_used = (reference_mask != 0) & (reference_mask == osm_mask) & (reference_mask != nn_mask)
            source_map[osm_used] = [0, 0, 255]
            
            both_used = (reference_mask != 0) & (reference_mask == nn_mask) & (reference_mask == osm_mask)
            source_map[both_used] = [128, 0, 128]
            
            background = (reference_mask == 0)
            source_map[background] = [50, 50, 50]
            
            axes[3].imshow(source_map)
            total = reference_mask.size
            osm_percent = (np.sum(osm_used) / total) * 100
            nn_percent = (np.sum(nn_used) / total) * 100
            axes[3].set_title(f'(d) Data Source\nOSM: {osm_percent:.1f}%, NN: {nn_percent:.1f}%', 
                             fontsize=12)
            axes[3].axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=np.array(self.CLASS_COLORS_RGB[0])/255, edgecolor='black', 
                  label='0: Background'),
            Patch(facecolor=np.array(self.CLASS_COLORS_RGB[1])/255, edgecolor='black', 
                  label='1: Buildings'),
            Patch(facecolor=np.array(self.CLASS_COLORS_RGB[2])/255, edgecolor='black', 
                  label='2: Woodlands'),
            Patch(facecolor=np.array(self.CLASS_COLORS_RGB[3])/255, edgecolor='black', 
                  label='3: Water'),
            Patch(facecolor=np.array(self.CLASS_COLORS_RGB[4])/255, edgecolor='black', 
                  label='4: Roads')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
                  fontsize=12, frameon=True, fancybox=True, shadow=True,
                  bbox_to_anchor=(0.5, -0.01))

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
            print(f"\nSaved visualization to: {save_path}")
        
        plt.close(fig)
    
    def process_coordinate_folder(self, coord_folder, strategy='thesis', min_area=50, save_results=True):
        print(f"\n{'='*70}")
        print(f"Creating Reference Map for: {coord_folder}")
        print(f"Strategy: {strategy}")
        print(f"Min area threshold: {min_area} pixels")
        print(f"{'='*70}")
        
        nn_mask, osm_mask, orto_image = self.load_masks(coord_folder)
        
        self.print_statistics(osm_mask, "OSM Topographic Map")
        self.print_statistics(nn_mask, "Neural Network Segmentation")
        
        print(f"\nCreating reference map with '{strategy}' strategy...")
        reference_mask = self.create_reference_map(nn_mask, osm_mask, strategy, min_area)
        
        self.print_statistics(reference_mask, "Reference Map (Merged)")
        
        if save_results:
            print("\nSaving results...")
            
            ref_mask_path = os.path.join(coord_folder, "reference_map.png")
            cv2.imwrite(ref_mask_path, reference_mask)
            print(f"  Saved reference mask: {ref_mask_path}")
            
            ref_colored = self.create_colored_mask(reference_mask)
            ref_colored_path = os.path.join(coord_folder, "reference_map_colored.png")
            cv2.imwrite(ref_colored_path, cv2.cvtColor(ref_colored, cv2.COLOR_RGB2BGR))
            print(f"  Saved colored reference: {ref_colored_path}")
            
            viz_path = os.path.join(coord_folder, "reference_map_comparison.png")
            self.visualize_comparison(nn_mask, osm_mask, reference_mask, orto_image, viz_path)
            
            metadata_path = os.path.join(coord_folder, "reference_map_metadata.txt")
            with open(metadata_path, 'w') as f:
                f.write(f"Reference Map Metadata\n")
                f.write(f"Coordinate Folder: {coord_folder}\n")
                f.write(f"Merge Strategy: {strategy}\n")
                f.write(f"Min Area Threshold: {min_area} pixels\n")
                f.write(f"Image Size: {reference_mask.shape}\n")
                f.write(f"\nFusion Hierarchy:\n")
                f.write(f"  1. OSM - Water bodies (most stable)\n")
                f.write(f"  2. OSM - Roads (relatively stable)\n")
                f.write(f"  3. NN Segmentation - Forests/Vegetation\n")
                f.write(f"  4. OSM + NN - Buildings (only OSM-confirmed)\n")
                f.write(f"  5. NN Segmentation - Background\n")
                f.write(f"\nClass Distribution:\n")
                unique, counts = np.unique(reference_mask, return_counts=True)
                total = reference_mask.size
                for class_id, count in zip(unique, counts):
                    class_name = self.CLASS_MAPPING.get(class_id, f'Unknown_{class_id}')
                    percentage = (count / total) * 100
                    f.write(f"  {class_name}: {count} pixels ({percentage:.2f}%)\n")
            
            print(f"  Saved metadata: {metadata_path}")
        
        print("Reference map creation complete!")
        
        return reference_mask


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        coord_folder = sys.argv[1]
        strategy = sys.argv[2] if len(sys.argv) > 2 else 'thesis'
        min_area = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        
        if not os.path.exists(coord_folder):
            print(f"ERROR: Folder not found: {coord_folder}")
            sys.exit(1)
        
        creator = ReferenceMapCreator()
        reference_mask = creator.process_coordinate_folder(
            coord_folder=coord_folder,
            strategy=strategy,
            min_area=min_area,
            save_results=True
        )
        
        print("\nAvailable strategies: thesis, hybrid, osm_priority, nn_priority, vote")
    else:
        print("Usage: python merge_topo_nn_cli.py <coord_folder> [strategy] [min_area]")
        print("\nExample:")
        print("  python merge_topo_nn_cli.py lat_54_371503_lon_18_618262 thesis 50")
        sys.exit(1)
