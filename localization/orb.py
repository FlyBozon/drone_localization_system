import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import json
import csv
import time
import math
from collections import deque
import pandas as pd

MAX_FRAMES = None  # None = wszystkie klatki
N_INIT_FRAMES = 10  # liczba początkowych klatek do kalibracji
HEADING_WINDOW = 5  # okno do uśredniania kierunku
N_HEADING_CORRECTION = 10  # liczba klatek GT do obliczenia kierunku po zakręcie
N_RETROSPECTIVE_CORRECTION = 10  # Co ile klatek robić retrospektywną korektę

# Ścieżki
img_dir = "datasets/uav-visloc/03/drone"
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
output = "output"

if MAX_FRAMES is not None and MAX_FRAMES < len(img_paths):
    img_paths = img_paths[:MAX_FRAMES]
    print(f"Ograniczono do pierwszych {MAX_FRAMES} klatek")
else:
    print(f"Przetwarzanie wszystkich {len(img_paths)} klatek")

print(f"Znaleziono obrazow: {len(img_paths)}")
if len(img_paths) == 0:
    print("Brak obrazow")
    exit()

# Tworzenie katalogów
save_dir = f"{output}/orb_localization_comparison"
os.makedirs(save_dir, exist_ok=True)
debug_dir = os.path.join(save_dir, "debug")
os.makedirs(debug_dir, exist_ok=True)
plots_dir = os.path.join(save_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Wczytaj ground truth csv
gt_csv = "datasets/uav-visloc/03/03.csv"
gt_list = []
gt_map = {}
with open(gt_csv, "r") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        fname = row[1]
        lat = float(row[3])
        lon = float(row[4])
        gt_map[fname] = {"lat": lat, "lon": lon}
        gt_list.append({"file": fname, "lat": lat, "lon": lon})

R_earth = 6378137.0

def latlon_to_enu(lat0, lon0, lat, lon):
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = R_earth * dlon * math.cos(math.radians(lat0))
    y = R_earth * dlat
    return x, y

def is_consistent(vec, mean, angle_thresh_deg=30):
    if np.linalg.norm(mean) < 1e-6:
        return True
    dot = np.dot(vec, mean)
    denom = (np.linalg.norm(vec) * np.linalg.norm(mean)) + 1e-9
    cosang = np.clip(dot / denom, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    return abs(angle) < angle_thresh_deg

def rotation_matrix_2d(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])

def get_heading_from_vector(vec):
    return np.degrees(np.arctan2(vec[1], vec[0]))

def normalize_angle(angle_deg):
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg

def circular_mean_angles(angles_deg):
    angles_rad = np.radians(angles_deg)
    x = np.mean(np.cos(angles_rad))
    y = np.mean(np.sin(angles_rad))
    return np.degrees(np.arctan2(y, x))

print("\nPRZED ROZPOCZĘCIEM LOTU: PRE-ANALIZA GROUND TRUTH")

if len(gt_list) == 0:
    print("Brak danych GT!")
    exit()

gt0 = gt_list[0]
lat0, lon0 = gt0["lat"], gt0["lon"]

print("\n1. Konwersja GT do układu ENU...")
gt_direct = []
for point in gt_list:
    x, y = latlon_to_enu(lat0, lon0, point["lat"], point["lon"])
    gt_direct.append([x, y])
gt_direct = np.array(gt_direct)
print(f"   Zbudowano trajektorię GT: {len(gt_direct)} punktów")

print("\n2. Obliczanie przemieszczeń między kolejnymi klatkami...")
displacements_gt = gt_direct[1:] - gt_direct[:-1]
distances_gt = np.linalg.norm(displacements_gt, axis=1)
headings_gt = np.degrees(np.arctan2(displacements_gt[:, 1], displacements_gt[:, 0]))

print(f"   Średnie przemieszczenie: {np.mean(distances_gt):.2f} m")
print(f"   Min przemieszczenie: {np.min(distances_gt):.2f} m")
print(f"   Max przemieszczenie: {np.max(distances_gt):.2f} m")

# Wykrywanie zakrętów
print("\n3. Wykrywanie zakrętów (~80° zmiana kierunku)...")
heading_changes = np.diff(headings_gt)
heading_changes = (heading_changes + 180) % 360 - 180

TURN_MIN = 50.0
TURN_MAX = 150.0

all_turn_indices = np.where(
    (np.abs(heading_changes) > TURN_MIN) &
    (np.abs(heading_changes) < TURN_MAX)
)[0] + 1

print(f"   Wykryto {len(all_turn_indices)} zakrętów")

# Grupowanie w pary
print("\n4. Grupowanie zakrętów w pary...")
turn_pairs = []
i = 0

while i < len(all_turn_indices):
    first_idx = all_turn_indices[i]
    
    if i + 1 < len(all_turn_indices) and all_turn_indices[i + 1] == first_idx + 1:
        second_idx = all_turn_indices[i + 1]
        displacement_at_first = gt_direct[second_idx] - gt_direct[first_idx- 1]
        distance_at_first = distances_gt[first_idx- 1]
        heading_before_first = headings_gt[first_idx - 1]
        heading_after_second = headings_gt[second_idx]
        rotation_angle = heading_changes[second_idx - 1]
        
        turn_pairs.append({
            "pair_id": len(turn_pairs),
            "first_turn_idx": int(first_idx),
            "second_turn_idx": int(second_idx),
            "lateral_displacement": displacement_at_first,
            "lateral_distance": float(distance_at_first),
            "heading_before": float(heading_before_first),
            "heading_after_rotation": float(heading_after_second),
            "rotation_angle": float(rotation_angle)
        })
        
        i += 2
    else:
        displacement_at_turn = displacements_gt[first_idx - 1]
        distance_at_turn = distances_gt[first_idx - 1]
        heading_before = headings_gt[first_idx - 1]
        heading_after = headings_gt[first_idx]
        rotation = heading_changes[first_idx - 1]
        
        turn_pairs.append({
            "pair_id": len(turn_pairs),
            "first_turn_idx": int(first_idx),
            "second_turn_idx": None,
            "lateral_displacement": displacement_at_turn,
            "lateral_distance": float(distance_at_turn),
            "heading_before": float(heading_before),
            "heading_after_rotation": float(heading_after),
            "rotation_angle": float(rotation)
        })
        
        i += 1

first_turn_indices = {pair["first_turn_idx"]: pair for pair in turn_pairs}
second_turn_indices = {pair["second_turn_idx"]: pair for pair in turn_pairs if pair["second_turn_idx"] is not None}

# Początkowy kierunek
print("\n5. Początkowy kierunek lotu...")
initial_direction_vector = np.mean(displacements_gt[:N_INIT_FRAMES], axis=0)
initial_flight_heading = get_heading_from_vector(initial_direction_vector)
print(f"   Początkowy heading: {initial_flight_heading:.1f}°")

print("\nPRE-ANALIZA ZAKOŃCZONA")
print(f"Wykrytych par zakrętów: {len(turn_pairs)}")

# INICJALIZACJA ORB
orb = cv2.ORB_create(
    nfeatures=4000,
    scaleFactor=1.01,
    nlevels=12,
    edgeThreshold=31,
    patchSize=31
)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# KROK 1: KALIBRACJA
print("KROK 1: KALIBRACJA - WYZNACZENIE SKALI I ROTACJI")

calibration_data = []
orb_displacements_raw = []
gt_displacements = []

n_calib = min(N_INIT_FRAMES, len(img_paths))

prev_path = img_paths[0]
prev_name = os.path.basename(prev_path)
prev_img = cv2.imread(prev_path)
prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

calib_start_time = time.time()

for i in range(1, n_calib):
    img_path = img_paths[i]
    img_name = os.path.basename(img_path)
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    
    matches = bf.match(prev_des, des)
    matches = sorted(matches, key=lambda x: x.distance)
    
    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]) if len(matches) > 0 else np.empty((0,2))
    pts2 = np.float32([kp[m.trainIdx].pt for m in matches]) if len(matches) > 0 else np.empty((0,2))
    
    if len(matches) >= 4:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        matches_ransac = [m for m, keep in zip(matches, mask.ravel()) if keep == 1] if mask is not None else []
    else:
        matches_ransac = []
    
    pts1_r = np.float32([prev_kp[m.queryIdx].pt for m in matches_ransac]) if len(matches_ransac) > 0 else np.empty((0,2))
    pts2_r = np.float32([kp[m.trainIdx].pt for m in matches_ransac]) if len(matches_ransac) > 0 else np.empty((0,2))
    flow = pts2_r - pts1_r if pts1_r.size else np.empty((0,2))
    mean_flow = np.mean(flow, axis=0) if flow.size else np.array([0.0, 0.0])
    
    matches_final = []
    for m in matches_ransac:
        v = np.array(kp[m.trainIdx].pt) - np.array(prev_kp[m.queryIdx].pt)
        if flow.size == 0 or is_consistent(v, mean_flow):
            matches_final.append(m)
    
    flow_final = np.array([np.array(kp[m.trainIdx].pt) - np.array(prev_kp[m.queryIdx].pt)
                           for m in matches_final]) if len(matches_final) > 0 else np.empty((0,2))
    
    if flow_final.size > 0:
        orb_displacement_raw = np.mean(flow_final, axis=0)
        orb_dist_px = float(np.linalg.norm(orb_displacement_raw))
    else:
        orb_displacement_raw = np.array([0.0, 0.0])
        orb_dist_px = 0.0
    
    if prev_name in gt_map and img_name in gt_map:
        lat1, lon1 = gt_map[prev_name]["lat"], gt_map[prev_name]["lon"]
        lat2, lon2 = gt_map[img_name]["lat"], gt_map[img_name]["lon"]
        
        x1, y1 = latlon_to_enu(lat0, lon0, lat1, lon1)
        x2, y2 = latlon_to_enu(lat0, lon0, lat2, lon2)
        
        gt_displacement = np.array([x2 - x1, y2 - y1])
        gt_dist_m = float(np.linalg.norm(gt_displacement))
    else:
        gt_displacement = np.array([0.0, 0.0])
        gt_dist_m = 0.0
    
    orb_displacements_raw.append(orb_displacement_raw)
    gt_displacements.append(gt_displacement)
    
    prev_img = img
    prev_gray = gray
    prev_kp, prev_des = kp, des
    prev_name = img_name

calib_time = time.time() - calib_start_time

orb_displacements_raw = np.array(orb_displacements_raw)
gt_displacements = np.array(gt_displacements)

orb_magnitudes = np.linalg.norm(orb_displacements_raw, axis=1)
gt_magnitudes = np.linalg.norm(gt_displacements, axis=1)

valid_mask = (orb_magnitudes > 1e-6) & (gt_magnitudes > 1e-6)
if np.sum(valid_mask) > 0:
    scale_estimates = gt_magnitudes[valid_mask] / orb_magnitudes[valid_mask]
    scale_factor = np.median(scale_estimates)
    scale_std = np.std(scale_estimates)
else:
    scale_factor = 1.0
    scale_std = 0.0

orb_angles = np.arctan2(orb_displacements_raw[:, 1], orb_displacements_raw[:, 0])
gt_angles = np.arctan2(gt_displacements[:, 1], gt_displacements[:, 0])

angle_diffs = gt_angles - orb_angles
angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))

initial_rotation_angle = np.median(angle_diffs[valid_mask]) if np.sum(valid_mask) > 0 else 0.0
initial_rotation_matrix = rotation_matrix_2d(initial_rotation_angle)

print("\nPARAMETRY KALIBRACJI:")
print(f"Czas kalibracji: {calib_time:.2f} s")
print(f"Liczba klatek: {n_calib - 1}")
print(f"SKALA (m/px): {scale_factor:.6f} ± {scale_std:.6f}")
print(f"ROTACJA: {np.degrees(initial_rotation_angle):.2f}°")
print(f"POCZĄTKOWY HEADING: {initial_flight_heading:.1f}°")

# FUNKCJA DO LOKALIZACJI (wspólna dla obu przypadków)
def run_localization(use_retrospective_correction=False):
    """
    Uruchom proces lokalizacji z lub bez retrospektywnej korekty
    
    Returns:
        trajectory_m: lista pozycji [(x, y), ...]
        error_history: lista błędów [error1, error2, ...]
        distance_km: lista przebytej drogi w km
        correction_frames: lista klatek z korektą
    """
    cumulative_distance_km = [0.0]
    pos_m = np.array([0.0, 0.0])
    trajectory_m = [(0.0, 0.0)]
    displacements_history = []
    error_history = []
    correction_frames = []
    
    displacement_history = deque(maxlen=HEADING_WINDOW)
    current_heading = initial_flight_heading
    
    prev_path = img_paths[0]
    prev_name = os.path.basename(prev_path)
    prev_img = cv2.imread(prev_path)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    
    for i in range(1, len(img_paths)):
        img_path = img_paths[i]
        img_name = os.path.basename(img_path)
        
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]) if len(matches) > 0 else np.empty((0,2))
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches]) if len(matches) > 0 else np.empty((0,2))
        
        if len(matches) >= 4:
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            matches_ransac = [m for m, keep in zip(matches, mask.ravel()) if keep == 1] if mask is not None else []
        else:
            matches_ransac = []
        
        pts1_r = np.float32([prev_kp[m.queryIdx].pt for m in matches_ransac]) if len(matches_ransac) > 0 else np.empty((0,2))
        pts2_r = np.float32([kp[m.trainIdx].pt for m in matches_ransac]) if len(matches_ransac) > 0 else np.empty((0,2))
        flow = pts2_r - pts1_r if pts1_r.size else np.empty((0,2))
        mean_flow = np.mean(flow, axis=0) if flow.size else np.array([0.0, 0.0])
        
        matches_final = []
        for m in matches_ransac:
            v = np.array(kp[m.trainIdx].pt) - np.array(prev_kp[m.queryIdx].pt)
            if flow.size == 0 or is_consistent(v, mean_flow):
                matches_final.append(m)
        
        flow_final = np.array([np.array(kp[m.trainIdx].pt) - np.array(prev_kp[m.queryIdx].pt)
                               for m in matches_final]) if len(matches_final) > 0 else np.empty((0,2))
        
        if flow_final.size > 0:
            orb_displacement_px = np.mean(flow_final, axis=0)
        else:
            orb_displacement_px = np.array([0.0, 0.0])
        
        # Sprawdź zakręty
        is_first_turn = i in first_turn_indices
        is_second_turn = i in second_turn_indices
        
        if is_first_turn:
            pair_info = first_turn_indices[i]
            lateral_displacement = pair_info["lateral_displacement"]
            actual_displacement_m = lateral_displacement.copy()
            
            if np.linalg.norm(actual_displacement_m) > 1e-6:
                displacement_history.append(actual_displacement_m)
            
            correction_frames.append(i)
        
        elif is_second_turn:
            pair_info = second_turn_indices[i]
            
            start_idx = i
            end_idx = min(i + N_HEADING_CORRECTION, len(displacements_gt))
            
            if end_idx > start_idx:
                future_displacements = displacements_gt[start_idx:end_idx]
                future_direction_vector = np.mean(future_displacements, axis=0)
                
                if np.linalg.norm(future_direction_vector) > 1e-6:
                    current_heading = get_heading_from_vector(future_direction_vector)
                else:
                    current_heading = pair_info["heading_after_rotation"]
            else:
                current_heading = pair_info["heading_after_rotation"]
            
            displacement_history.clear()
            actual_displacement_m = np.array([0.0, 0.0])
            correction_frames.append(i)
        
        else:
            # Normalna lokalizacja ORB
            if i < n_calib:
                if prev_name in gt_map and img_name in gt_map:
                    lat1, lon1 = gt_map[prev_name]["lat"], gt_map[prev_name]["lon"]
                    lat2, lon2 = gt_map[img_name]["lat"], gt_map[img_name]["lon"]
                    x1, y1 = latlon_to_enu(lat0, lon0, lat1, lon1)
                    x2, y2 = latlon_to_enu(lat0, lon0, lat2, lon2)
                    actual_displacement_m = np.array([x2 - x1, y2 - y1])
                else:
                    orb_displacement_rotated = initial_rotation_matrix @ orb_displacement_px
                    actual_displacement_m = scale_factor * orb_displacement_rotated
                
                if np.linalg.norm(actual_displacement_m) > 1e-6:
                    displacement_history.append(actual_displacement_m)
            else:
                orb_displacement_rotated = initial_rotation_matrix @ orb_displacement_px
                orb_displacement_scaled = scale_factor * orb_displacement_rotated
                
                if len(displacement_history) > 0:
                    headings = [get_heading_from_vector(d) for d in displacement_history if np.linalg.norm(d) > 1e-6]
                    if len(headings) > 0:
                        current_heading = circular_mean_angles(headings)
                
                if np.linalg.norm(orb_displacement_scaled) > 1e-6:
                    orb_current_heading = get_heading_from_vector(orb_displacement_scaled)
                    additional_rotation_deg = current_heading - orb_current_heading
                    additional_rotation_rad = np.radians(additional_rotation_deg)
                    additional_rotation_matrix = rotation_matrix_2d(additional_rotation_rad)
                    actual_displacement_m = additional_rotation_matrix @ orb_displacement_scaled
                else:
                    actual_displacement_m = orb_displacement_scaled
                
                if np.linalg.norm(actual_displacement_m) > 1e-6:
                    displacement_history.append(actual_displacement_m)
        
        displacements_history.append(actual_displacement_m.copy())
        
        pos_m += actual_displacement_m
        step_distance = np.linalg.norm(actual_displacement_m)
        cumulative_distance_km.append(cumulative_distance_km[-1] + step_distance / 1000.0)
        
        trajectory_m.append((pos_m[0], pos_m[1]))
        
        # Oblicz błąd
        if img_name in gt_map:
            lat_curr, lon_curr = gt_map[img_name]["lat"], gt_map[img_name]["lon"]
            x_gt, y_gt = latlon_to_enu(lat0, lon0, lat_curr, lon_curr)
        else:
            x_gt, y_gt = 0.0, 0.0
        
        error = np.linalg.norm(np.array([pos_m[0], pos_m[1]]) - np.array([x_gt, y_gt]))
        error_history.append(error)
        
        # RETROSPEKTYWNA KOREKTA (tylko jeśli włączona)
        if use_retrospective_correction and i >= N_RETROSPECTIVE_CORRECTION and i % N_RETROSPECTIVE_CORRECTION == 0 and i >= n_calib:
            correction_idx = i - N_RETROSPECTIVE_CORRECTION
            correction_frame_name = os.path.basename(img_paths[correction_idx])
            
            if correction_frame_name in gt_map:
                lat_corr, lon_corr = gt_map[correction_frame_name]["lat"], gt_map[correction_frame_name]["lon"]
                x_gt_corr, y_gt_corr = latlon_to_enu(lat0, lon0, lat_corr, lon_corr)
                
                trajectory_m[correction_idx] = (x_gt_corr, y_gt_corr)
                
                for j in range(correction_idx + 1, i + 1):
                    displacement = displacements_history[j - 1]
                    prev_pos = np.array(trajectory_m[j - 1])
                    new_pos = prev_pos + displacement
                    trajectory_m[j] = (new_pos[0], new_pos[1])
                
                pos_m = np.array(trajectory_m[i])
                error = np.linalg.norm(pos_m - np.array([x_gt, y_gt]))
                error_history[-1] = error
                correction_frames.append(i)
        
        prev_img = img
        prev_gray = gray
        prev_kp, prev_des = kp, des
        prev_name = img_name
    
    return trajectory_m, error_history, cumulative_distance_km, correction_frames

# URUCHOM OBA WARIANTY
print("WARIANT 1: BEZ RETROSPEKTYWNEJ KOREKTY")
start_time = time.time()
traj_no_retro, errors_no_retro, dist_no_retro, corr_no_retro = run_localization(use_retrospective_correction=False)
time_no_retro = time.time() - start_time
print(f"Czas: {time_no_retro:.2f} s")
print(f"Klatek z korektą GT: {len(corr_no_retro)}")

print("WARIANT 2: Z RETROSPEKTYWNĄ KOREKTĄ")
start_time = time.time()
traj_with_retro, errors_with_retro, dist_with_retro, corr_with_retro = run_localization(use_retrospective_correction=True)
time_with_retro = time.time() - start_time
print(f"Czas: {time_with_retro:.2f} s")
print(f"Klatek z korektą GT: {len(corr_with_retro)}")

# Przygotuj GT trajectory
gt_trajectory_m = [(0.0, 0.0)]
for i in range(1, len(img_paths)):
    img_name = os.path.basename(img_paths[i])
    if img_name in gt_map:
        lat_curr, lon_curr = gt_map[img_name]["lat"], gt_map[img_name]["lon"]
        x_gt, y_gt = latlon_to_enu(lat0, lon0, lat_curr, lon_curr)
    else:
        x_gt, y_gt = gt_trajectory_m[-1]
    gt_trajectory_m.append((x_gt, y_gt))

# WYKRES PORÓWNAWCZY
print("Generowanie wykresu porownawczego")

traj_no_retro_arr = np.array(traj_no_retro)
traj_with_retro_arr = np.array(traj_with_retro)
gt_arr = np.array(gt_trajectory_m)

# Statystyki
mean_err_no_retro = np.mean(errors_no_retro)
max_err_no_retro = np.max(errors_no_retro)
final_err_no_retro = errors_no_retro[-1]
total_dist_km = dist_no_retro[-1]

mean_err_with_retro = np.mean(errors_with_retro)
max_err_with_retro = np.max(errors_with_retro)
final_err_with_retro = errors_with_retro[-1]

print(f"\nSTATYSTYKI PORÓWNAWCZE:")
print(f"Przebyta droga: {total_dist_km:.3f} km")
print(f"\nBEZ retrospektywnej korekty:")
print(f"  Średni błąd: {mean_err_no_retro:.2f} m")
print(f"  Maksymalny błąd: {max_err_no_retro:.2f} m")
print(f"  Końcowy błąd: {final_err_no_retro:.2f} m")
print(f"\nZ retrospektywną korektą:")
print(f"  Średni błąd: {mean_err_with_retro:.2f} m")
print(f"  Maksymalny błąd: {max_err_with_retro:.2f} m")
print(f"  Końcowy błąd: {final_err_with_retro:.2f} m")
print(f"\nPOPRAWA:")
print(f"  Średni błąd: {mean_err_no_retro - mean_err_with_retro:.2f} m ({(1 - mean_err_with_retro/mean_err_no_retro)*100:.1f}%)")
print(f"  Maksymalny błąd: {max_err_no_retro - max_err_with_retro:.2f} m ({(1 - max_err_with_retro/max_err_no_retro)*100:.1f}%)")
print(f"  Końcowy błąd: {final_err_no_retro - final_err_with_retro:.2f} m ({(1 - final_err_with_retro/final_err_no_retro)*100:.1f}%)")

# Stwórz wykres
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Wykres 1: Trajektorie
ax = axes[0]
ax.plot(gt_arr[:, 0], gt_arr[:, 1], 'g-', label='Ground Truth', 
        linewidth=2.5, alpha=0.9, zorder=1)
ax.plot(traj_no_retro_arr[:, 0], traj_no_retro_arr[:, 1], 'b-', 
        label=f'Bez korekty (śr. błąd: {mean_err_no_retro:.1f}m, max: {max_err_no_retro:.1f}m, końc: {final_err_no_retro:.1f}m)', 
        linewidth=2, alpha=0.7, zorder=2)
ax.plot(traj_with_retro_arr[:, 0], traj_with_retro_arr[:, 1], 'r-', 
        label=f'Z korektą co {N_RETROSPECTIVE_CORRECTION} klatki (śr. błąd: {mean_err_with_retro:.1f}m, max: {max_err_with_retro:.1f}m, końc: {final_err_with_retro:.1f}m)', 
        linewidth=2, alpha=0.7, zorder=3)

# ax.plot(traj_no_retro_arr[0, 0], traj_no_retro_arr[0, 1], 'ko', 
#         markersize=12, label='Start', zorder=4)
# ax.plot(traj_no_retro_arr[-1, 0], traj_no_retro_arr[-1, 1], 'k*', 
#         markersize=15, label='Koniec', zorder=4)

ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.set_xlabel("E [m]", fontsize=14)
ax.set_ylabel("N [m]", fontsize=14)
ax.set_title(f"Porównanie trajektorii\nPrzebyta droga: {total_dist_km:.3f} km", 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis("equal")

# Wykres 2: Błąd w czasie
ax = axes[1]
distances_km = dist_no_retro[1:]
ax.plot(distances_km, errors_no_retro, 'b-', linewidth=2, alpha=0.7, 
        label=f'Bez korekty')
ax.plot(distances_km, errors_with_retro, 'r-', linewidth=2, alpha=0.7, 
        label=f'Z korektą co {N_RETROSPECTIVE_CORRECTION} klatki')

# Zaznacz momenty korekty
for corr_frame in corr_with_retro:
    if corr_frame < len(distances_km):
        ax.axvline(x=distances_km[corr_frame-1], color='orange', 
                   linestyle='--', alpha=0.3, linewidth=1)

ax.set_xlabel("Przebyta droga [km]", fontsize=14)
ax.set_ylabel("Błąd [m]", fontsize=14)
ax.set_title(f"Błąd lokalizacji w funkcji drogi\nPoprawa średniego błędu: {(1 - mean_err_with_retro/mean_err_no_retro)*100:.1f}%", 
             fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
comparison_path = os.path.join(plots_dir, "comparison_trajectories.png")
plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
print(f"\nWykres porównawczy zapisany: {comparison_path}")
plt.close()

print("\nZapisywanie danych...")

# DataFrame dla obu wariantów
data_export = []
for i in range(len(traj_no_retro)):
    frame_name = os.path.basename(img_paths[i]) if i < len(img_paths) else f"frame_{i}"
    
    data_export.append({
        "frame": i,
        "filename": frame_name,
        "no_retro_x_m": traj_no_retro[i][0],
        "no_retro_y_m": traj_no_retro[i][1],
        "with_retro_x_m": traj_with_retro[i][0],
        "with_retro_y_m": traj_with_retro[i][1],
        "gt_x_m": gt_trajectory_m[i][0],
        "gt_y_m": gt_trajectory_m[i][1],
        "error_no_retro_m": errors_no_retro[i-1] if i > 0 else 0.0,
        "error_with_retro_m": errors_with_retro[i-1] if i > 0 else 0.0,
        "distance_km": dist_no_retro[i] if i < len(dist_no_retro) else 0.0
    })

# CSV
csv_path = os.path.join(save_dir, "comparison_data.csv")
df = pd.DataFrame(data_export)
df.to_csv(csv_path, index=False)
print(f"Zapisano CSV: {csv_path}")

# JSON
json_path = os.path.join(save_dir, "comparison_data.json")
with open(json_path, "w") as f:
    json.dump({
        "metadata": {
            "total_frames": len(traj_no_retro),
            "calibration_frames": n_calib,
            "turn_pairs": len(turn_pairs),
            "total_distance_km": total_dist_km,
            "retrospective_correction_interval": N_RETROSPECTIVE_CORRECTION,
            "no_retro": {
                "mean_error_m": float(mean_err_no_retro),
                "max_error_m": float(max_err_no_retro),
                "final_error_m": float(final_err_no_retro),
                "gt_corrections": len(corr_no_retro)
            },
            "with_retro": {
                "mean_error_m": float(mean_err_with_retro),
                "max_error_m": float(max_err_with_retro),
                "final_error_m": float(final_err_with_retro),
                "gt_corrections": len(corr_with_retro)
            },
            "improvement": {
                "mean_error_m": float(mean_err_no_retro - mean_err_with_retro),
                "mean_error_percent": float((1 - mean_err_with_retro/mean_err_no_retro)*100),
                "max_error_m": float(max_err_no_retro - max_err_with_retro),
                "max_error_percent": float((1 - max_err_with_retro/max_err_no_retro)*100)
            }
        },
        "trajectory": data_export
    }, f, indent=2)
print(f"Zapisano JSON: {json_path}")

print("ZAKOŃCZONO")