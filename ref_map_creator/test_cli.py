import osmnx as ox
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import requests
import math
import numpy as np
from PIL import Image
import io
import os
import sys
from datetime import datetime

import shapely.geometry as geom
from shapely.ops import unary_union
import geopandas as gpd
import osmnx as ox
import math
import pandas as pd

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.max_query_area_size = 50000 * 50000

class OrtoFetcher:
    def __init__(self):
        self.base_url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMTS/StandardResolution"
        self.tile_size = 256
    
    def lat_lon_to_tile(self, lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x_tile, y_tile
    
    def tile_to_lat_lon(self, x, y, zoom):
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return lat, lon
    
    def download_tile(self, x, y, zoom):
        url = f"{self.base_url}?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTOFOTOMAPA&TILEMATRIXSET=EPSG:3857&FORMAT=image/jpeg&TileMatrix={zoom}&TileRow={y}&TileCol={x}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"Error downloading tile {x},{y},{zoom}: {e}")
            return Image.new('RGB', (self.tile_size, self.tile_size), color='gray')
    
    def fetch_area(self, north, south, east, west, zoom=18):
        nw_tile_x, nw_tile_y = self.lat_lon_to_tile(north, west, zoom)
        se_tile_x, se_tile_y = self.lat_lon_to_tile(south, east, zoom)
        
        min_x = min(nw_tile_x, se_tile_x)
        max_x = max(nw_tile_x, se_tile_x)
        min_y = min(nw_tile_y, se_tile_y)
        max_y = max(nw_tile_y, se_tile_y)
        
        print(f"Downloading tiles: X[{min_x}:{max_x}], Y[{min_y}:{max_y}]")
        print(f"Total tiles: {(max_x - min_x + 1) * (max_y - min_y + 1)}")
        
        width = (max_x - min_x + 1) * self.tile_size
        height = (max_y - min_y + 1) * self.tile_size
        
        result_image = Image.new('RGB', (width, height))
        
        tile_count = 0
        total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile = self.download_tile(x, y, zoom)
                paste_x = (x - min_x) * self.tile_size
                paste_y = (y - min_y) * self.tile_size
                result_image.paste(tile, (paste_x, paste_y))
                tile_count += 1
                if tile_count % 10 == 0 or tile_count == total_tiles:
                    print(f"  Progress: {tile_count}/{total_tiles} tiles", end='\r', flush=True)
        
        print()  # New line after progress
        
        actual_north, actual_west = self.tile_to_lat_lon(min_x, min_y, zoom)
        actual_south, actual_east = self.tile_to_lat_lon(max_x + 1, max_y + 1, zoom)
        
        return result_image, (actual_north, actual_south, actual_east, actual_west)


def create_road_polygons(roads_gdf, default_width_meters=4):
    if roads_gdf.empty:
        return gpd.GeoDataFrame()
    
    ROAD_WIDTHS = {
        'motorway': 12, 'motorway_link': 8, 'trunk': 10, 'trunk_link': 8,
        'primary': 8, 'primary_link': 6, 'secondary': 7, 'secondary_link': 6,
        'tertiary': 6, 'tertiary_link': 5, 'residential': 5, 'living_street': 4,
        'unclassified': 4, 'service': 3, 'road': 4,
    }
    EXCLUDED_TYPES = {
        'footway', 'path', 'pedestrian', 'steps', 'bridleway', 
        'cycleway', 'track', 'corridor', 'sidewalk'
    }
    EXCLUDED_SURFACES = {
        'unpaved', 'gravel', 'dirt', 'sand', 'grass', 'ground',
        'mud', 'earth', 'grass_paver', 'woodchips', 'soil'
    }
    
    roads_copy = roads_gdf.copy()
    
    if 'highway' in roads_copy.columns:
        print(f"    Filtering roads by type (before: {len(roads_copy)} roads)...")
        roads_copy = roads_copy[~roads_copy['highway'].isin(EXCLUDED_TYPES)]
        print(f"    After type filter: {len(roads_copy)} roads")
    
    if 'surface' in roads_copy.columns:
        print(f"    Filtering roads by surface...")
        before_count = len(roads_copy)
        roads_copy = roads_copy[~roads_copy['surface'].isin(EXCLUDED_SURFACES)]
        print(f"    Removed {before_count - len(roads_copy)} unpaved roads")
    
    if roads_copy.empty:
        return gpd.GeoDataFrame()
    
    roads_copy = roads_copy[roads_copy.geometry.type.isin(['LineString', 'MultiLineString'])]
    
    if roads_copy.empty:
        return gpd.GeoDataFrame()
    
    def get_buffer_distance(row):
        highway_type = row.get('highway', 'unclassified')
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        
        width_meters = ROAD_WIDTHS.get(highway_type, default_width_meters)
        meters_per_degree = 80000
        buffer_degrees = width_meters / meters_per_degree
        
        return buffer_degrees
    
    print(f"    Creating road polygons with buffers...")
    
    roads_copy['buffer_dist'] = roads_copy.apply(get_buffer_distance, axis=1)
    roads_copy['geometry'] = roads_copy.apply(
        lambda row: row.geometry.buffer(row.buffer_dist), axis=1
    )
    
    roads_copy = roads_copy.drop(columns=['buffer_dist'])
    
    print(f"    Created {len(roads_copy)} road polygons")
    
    return roads_copy


def fetch_osm_data(north, south, east, west, buffer_factor=0.15):
    lat_range = north - south
    lon_range = east - west
    lat_buffer = lat_range * buffer_factor
    lon_buffer = lon_range * buffer_factor
    
    expanded_north = north + lat_buffer
    expanded_south = south - lat_buffer
    expanded_east = east + lon_buffer
    expanded_west = west - lon_buffer
    
    print(f"Fetching OSM data for bounds:")
    print(f"  Original: N={north:.6f}, S={south:.6f}, E={east:.6f}, W={west:.6f}")
    print(f"  Expanded: N={expanded_north:.6f}, S={expanded_south:.6f}, E={expanded_east:.6f}, W={expanded_west:.6f}")
    print(f"  Buffer: {buffer_factor*100:.1f}% expansion")

    poly = geom.box(expanded_west, expanded_south, expanded_east, expanded_north)

    lat_km = (expanded_north - expanded_south) * 111
    lon_km = (expanded_east - expanded_west) * 111 * math.cos(math.radians((expanded_north + expanded_south) / 2))
    print(f"  Area: ~{lat_km:.2f}km × {lon_km:.2f}km = {lat_km*lon_km:.2f} km²")

    data = {}

    try:
        print("  - Fetching buildings...", end="", flush=True)
        data["buildings"] = ox.features_from_polygon(poly, tags={"building": True})
        print(f" {len(data['buildings'])} found")
    except Exception as e:
        print(f" FAILED: {e}")
        data["buildings"] = gpd.GeoDataFrame()

    try:
        print("  - Fetching water bodies...", end="", flush=True)
        water_tags = {"natural": ["water", "bay"], "waterway": True}
        data["water"] = ox.features_from_polygon(poly, tags=water_tags)
        print(f" {len(data['water'])} found")
    except Exception as e:
        print(f" FAILED: {e}")
        data["water"] = gpd.GeoDataFrame()

    try:
        print("  - Fetching roads...", end="", flush=True)
        roads_tags = {"highway": True}
        roads_gdf = ox.features_from_polygon(poly, tags=roads_tags)
        data["roads"] = roads_gdf
        print(f" {len(roads_gdf)} found")
        
        print("  - Converting roads to polygons...")
        data["road_polygons"] = create_road_polygons(roads_gdf)
        print(f"    {len(data['road_polygons'])} road polygons created")
    except Exception as e:
        print(f" FAILED: {e}")
        data["roads"] = gpd.GeoDataFrame()
        data["road_polygons"] = gpd.GeoDataFrame()

    return data


def fetch_parks_and_green(poly):
    print("  - Fetching parks and green areas...", end="", flush=True)
    
    green_tags = {
        "landuse": ["forest", "grass", "meadow", "orchard", "vineyard", "farmland"],
        "natural": ["wood", "scrub", "heath", "grassland"],
        "leisure": ["park", "garden", "nature_reserve"]
    }
    
    try:
        parks_green = ox.features_from_polygon(poly, tags=green_tags)
        print(f" {len(parks_green)} found")
        return parks_green
    except Exception as e:
        print(f" FAILED: {e}")
        return gpd.GeoDataFrame()


def create_grayscale_segmentation_mask(osm_data, bounds, image_size):
    north, south, east, west = bounds
    width, height = image_size
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    def latlon_to_pixel(lat, lon):
        x = int((lon - west) / (east - west) * width)
        y = int((north - lat) / (north - south) * height)
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        return x, y
    
    def draw_geometry(geom, class_value):
        if geom.is_empty:
            return
        
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            pixels = np.array([latlon_to_pixel(lat, lon) for lon, lat in coords])
            if len(pixels) > 2:
                from PIL import Image, ImageDraw
                temp_img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(temp_img)
                draw.polygon([tuple(p) for p in pixels], fill=class_value)
                temp_mask = np.array(temp_img)
                mask[:] = np.maximum(mask, temp_mask)
        
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                draw_geometry(poly, class_value)
        
        elif geom.geom_type == 'LineString':
            coords = list(geom.coords)
            pixels = np.array([latlon_to_pixel(lat, lon) for lon, lat in coords])
            if len(pixels) > 1:
                import cv2
                for i in range(len(pixels) - 1):
                    cv2.line(mask, tuple(pixels[i]), tuple(pixels[i+1]), class_value, 2)
        
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                draw_geometry(line, class_value)
    
    print("  Drawing water (class 3)...")
    if not osm_data['water'].empty:
        for idx, row in osm_data['water'].iterrows():
            draw_geometry(row.geometry, 3)
    
    print("  Drawing vegetation (class 2)...")
    if 'parks_green' in osm_data and not osm_data['parks_green'].empty:
        for idx, row in osm_data['parks_green'].iterrows():
            draw_geometry(row.geometry, 2)
    
    print("  Drawing roads (class 4)...")
    if not osm_data['road_polygons'].empty:
        for idx, row in osm_data['road_polygons'].iterrows():
            draw_geometry(row.geometry, 4)
    
    print("  Drawing buildings (class 1)...")
    if not osm_data['buildings'].empty:
        for idx, row in osm_data['buildings'].iterrows():
            draw_geometry(row.geometry, 1)
    
    return mask


def test_alignment(center_lat, center_lon, size_deg=0.003, zoom=18, osm_buffer=0.15):
    print(f"\n{'='*70}")
    print(f"Starting test_alignment")
    print(f"{'='*70}")
    print(f"Center: ({center_lat}, {center_lon})")
    print(f"Size: {size_deg}° × {size_deg}°")
    print(f"Zoom: {zoom}")
    print(f"OSM Buffer: {osm_buffer*100:.1f}%")
    print()
    
    half_size = size_deg / 2
    north = center_lat + half_size
    south = center_lat - half_size
    east = center_lon + half_size
    west = center_lon - half_size
    
    # Folder name based on center coordinates
    folder_name = f"lat_{center_lat:.6f}_lon_{center_lon:.6f}".replace('.', '_').replace('-', 'neg')
    output_dir = folder_name
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}/")
    print()
    
    # Fetch orthophoto
    print("1. Downloading orthophoto from Geoportal...")
    fetcher = OrtoFetcher()
    orto_image, actual_bounds = fetcher.fetch_area(north, south, east, west, zoom)
    actual_north, actual_south, actual_east, actual_west = actual_bounds
    
    # Save orthophoto
    orto_filename = os.path.join(output_dir, "image_orto.jpg")
    orto_image.save(orto_filename, "JPEG", quality=95)
    print(f"  Saved orthophoto: {orto_filename}")
    print(f"  Image size: {orto_image.size[0]}×{orto_image.size[1]} pixels")
    
    # Calculate resolution
    lat_diff_km = (actual_north - actual_south) * 111
    lon_diff_km = (actual_east - actual_west) * 111 * math.cos(math.radians(center_lat))
    meters_per_pixel_lat = (lat_diff_km * 1000) / orto_image.size[1]
    meters_per_pixel_lon = (lon_diff_km * 1000) / orto_image.size[0]
    print(f"  Resolution: ~{meters_per_pixel_lat:.2f}m/pixel (lat), ~{meters_per_pixel_lon:.2f}m/pixel (lon)")
    print()
    
    # Fetch OSM data
    print("2. Fetching OSM topographic data...")
    osm_data = fetch_osm_data(actual_north, actual_south, actual_east, actual_west, osm_buffer)
    print()
    
    # Fetch parks/green areas
    lat_range = actual_north - actual_south
    lon_range = actual_east - actual_west
    lat_buffer = lat_range * osm_buffer
    lon_buffer = lon_range * osm_buffer
    
    expanded_north = actual_north + lat_buffer
    expanded_south = actual_south - lat_buffer
    expanded_east = actual_east + lon_buffer
    expanded_west = actual_west - lon_buffer
    
    poly = geom.box(expanded_west, expanded_south, expanded_east, expanded_north)
    osm_data["parks_green"] = fetch_parks_and_green(poly)
    
    osm_data["vegetation"] = osm_data["parks_green"]
    osm_data["forests"] = osm_data["parks_green"]
    
    print("3. Creating grayscale segmentation mask...")
    seg_mask = create_grayscale_segmentation_mask(osm_data, actual_bounds, orto_image.size)
    
    # Save segmentation mask
    seg_mask_filename = os.path.join(output_dir, "segmentation_mask.png")
    Image.fromarray(seg_mask, mode='L').save(seg_mask_filename)
    print(f"  Saved segmentation mask: {seg_mask_filename}")
    print(f"  Unique classes in mask: {np.unique(seg_mask)}")
    print()
    
    print("4. Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))
    axes = axes.flatten()
    
    print("  - Plotting orthophoto...")
    axes[0].imshow(orto_image, extent=[actual_west, actual_east, actual_south, actual_north],
                   aspect='auto')
    axes[0].set_title('Orthophoto')
    axes[0].set_xlabel('Lon')
    axes[0].set_ylabel('Lat')
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0].tick_params(labelsize=8)
    
    print("  - Plotting OSM features...")
    axes[1].set_xlim(actual_west, actual_east)
    axes[1].set_ylim(actual_south, actual_north)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_facecolor('#f5f5f5')
    
    if not osm_data['water'].empty:
        try:
            osm_data['water'].plot(ax=axes[1], color='#87CEEB', 
                                   edgecolor='#4682B4', linewidth=0.5, alpha=1.0)
            print("    Water plotted")
        except Exception as e:
            print(f"    Could not plot water: {e}")
    
    if not osm_data['forests'].empty:
        try:
            osm_data['forests'].plot(ax=axes[1], color='#90EE90', 
                                     edgecolor='#228B22', linewidth=0.5, alpha=1.0)
            print("    Forests plotted")
        except Exception as e:
            print(f"    Could not plot forests: {e}")
    
    if not osm_data['road_polygons'].empty:
        try:
            osm_data['road_polygons'].plot(ax=axes[1], color='#696969', 
                                           edgecolor='#404040', linewidth=0.3, alpha=1.0)
            print("    Road polygons plotted")
        except Exception as e:
            print(f"    Could not plot road polygons: {e}")
    
    if not osm_data['buildings'].empty:
        try:
            osm_data['buildings'].plot(ax=axes[1], color='#D2691E', 
                                       edgecolor='#8B4513', linewidth=0.5, alpha=1.0)
            print("    Buildings plotted")
        except Exception as e:
            print(f"    Could not plot buildings: {e}")
    
    axes[1].set_title('OSM')
    axes[1].set_xlabel('Lon')
    axes[1].set_ylabel('Lat')
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[1].tick_params(labelsize=8)
    
    print("  - Creating overlay...")
    axes[2].imshow(orto_image, extent=[actual_west, actual_east, actual_south, actual_north],
                   aspect='auto', alpha=1.0)
    axes[2].set_xlim(actual_west, actual_east)
    axes[2].set_ylim(actual_south, actual_north)
    
    if not osm_data['road_polygons'].empty:
        try:
            osm_data['road_polygons'].plot(ax=axes[2], color='#FFFF00', 
                                           edgecolor='#FFD700', linewidth=0.5, alpha=0.6)
            print("    Road polygons overlaid")
        except Exception as e:
            print(f"    Could not overlay road polygons: {e}")
    
    if not osm_data['buildings'].empty:
        try:
            osm_data['buildings'].plot(ax=axes[2], facecolor='none', 
                                       edgecolor='#FF0000', linewidth=2, alpha=1.0)
            print("    Buildings overlaid")
        except Exception as e:
            print(f"    Could not overlay buildings: {e}")
    
    if not osm_data['water'].empty:
        try:
            osm_data['water'].plot(ax=axes[2], facecolor='none', 
                                   edgecolor='#00FFFF', linewidth=2, alpha=1.0)
            print("    Water overlaid")
        except Exception as e:
            print(f"    Could not overlay water: {e}")
    
    if not osm_data['forests'].empty:
        try:
            osm_data['forests'].plot(ax=axes[2], facecolor='none', 
                                     edgecolor='#00FF00', linewidth=2, alpha=1.0)
            print("    Forests overlaid")
        except Exception as e:
            print(f"    Could not overlay forests: {e}")
    
    axes[2].set_title('Połączenie (OSM na Orthophoto)')
    axes[2].set_xlabel('Lon')
    axes[2].set_ylabel('Lat')
    axes[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[2].tick_params(labelsize=8)
    
    print("  - Plotting grayscale segmentation mask...")
    axes[3].imshow(seg_mask, extent=[actual_west, actual_east, actual_south, actual_north],
                   cmap='gray', vmin=0, vmax=4, aspect='auto', interpolation='nearest')
    axes[3].set_xlim(actual_west, actual_east)
    axes[3].set_ylim(actual_south, actual_north)
    
    axes[3].set_title('Grayscale Segmentation Mask')
    axes[3].set_xlabel('Lon')
    axes[3].set_ylabel('Lat')
    axes[3].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='white')
    axes[3].tick_params(labelsize=8)
    
    from matplotlib.patches import Patch
    seg_legend = [
        Patch(facecolor='#000000', label='0 - Unlabeled/Tło'),
        Patch(facecolor='#333333', label='1 - Buildings/Budynki'),
        Patch(facecolor='#666666', label='2 - Woodlands/Lasy'),
        Patch(facecolor='#999999', label='3 - Water/Woda'),
        Patch(facecolor='#CCCCCC', label='4 - Roads/Drogi')
    ]
    axes[3].legend(handles=seg_legend, loc='upper right', fontsize=7)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='#FF0000', linewidth=2, label='Budynki'),
        Patch(facecolor='#FFFF00', alpha=0.6, label='Drogi'),
        Patch(facecolor='none', edgecolor='#00FFFF', linewidth=2, label='Woda'),
        Patch(facecolor='none', edgecolor='#00FF00', linewidth=2, label='Roślinność')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'alignment_visualization.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    
    plt.close(fig)
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"Center: ({center_lat}, {center_lon})\n")
        f.write(f"Size: {size_deg}° × {size_deg}°\n")
        f.write(f"Zoom: {zoom}\n")
        f.write(f"OSM Buffer: {osm_buffer*100:.1f}%\n")
        f.write(f"Bounds: N={actual_north:.6f}, S={actual_south:.6f}, E={actual_east:.6f}, W={actual_west:.6f}\n")
        f.write(f"Image size: {orto_image.size[0]}×{orto_image.size[1]} pixels\n")
        f.write(f"Resolution: ~{meters_per_pixel_lat:.2f}m/pixel\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    print(f"Saved metadata to: {metadata_file}")
    
    print(f"\nAll files saved in directory: {output_dir}/")
    print(f"  - image_orto.jpg (orthophoto)")
    print(f"  - segmentation_mask.png (grayscale mask with class values 0-4)")
    print(f"  - alignment_visualization.png (visualization)")
    print(f"  - metadata.txt (metadata)")
    
    return orto_image, osm_data, actual_bounds


if __name__ == '__main__':
    if len(sys.argv) >= 5:
        # CLI mode
        center_lat = float(sys.argv[1])
        center_lon = float(sys.argv[2])
        size_deg = float(sys.argv[3])
        zoom = int(sys.argv[4])
        osm_buffer = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
        
        test_alignment(center_lat, center_lon, size_deg, zoom, osm_buffer)
    else:
        # Default example
        print("Usage: python test.py <center_lat> <center_lon> <size_deg> <zoom> [osm_buffer]")
        print("Running with default parameters (Gdańsk)...")
        test_alignment(54.352, 18.646, size_deg=0.003, zoom=18, osm_buffer=0.15)
