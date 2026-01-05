#!/usr/bin/env python3

import os
import glob
import re
import cv2
import numpy as np
import math
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
import json

#oryginally
DATA_FOLDER ="output/dataset"
DRONE_NAME = "02_0007"
PATCH_GLOB = os.path.join(DATA_FOLDER, DRONE_NAME, "*_row*_col*_mask.png")
REF_MAP_PATH = os.path.join(DATA_FOLDER, DRONE_NAME, "ref_map.png")

# Matching hyperparams
MAX_MATCH_DIST = 40
NEIGHBOR_RADIUS = 200
MIN_MATCHED = 2
VOTE_BIN = 15
RANSAC_ITERS = 600
RANSAC_MIN_INLIERS = 2
DOWNSAMPLE_SCALE = 0.25
FALLBACK_TM_WEIGHT = 0.6

# Class weights
CLASS_WEIGHT_DEFAULT = 0.6
CLASS_WEIGHTS = {
    0: 0.2, 1: 1.0, 2: 0.5, 3: 0.7, 4: 0.9
}

# Methods to run
METHODS_TO_RUN = [
    "voting_accumulator",
    "ransac_translation",
    "combined_voting_ransac",
]

# Polish names
METHODS_POLISH_NAMES = {
    "voting_accumulator": "Akumulator głosowań",
    "ransac_translation": "RANSAC translacja",
    "combined_voting_ransac": "Połączone głosowanie + RANSAC",
}

# Output
OUT_DIR = os.path.join(DATA_FOLDER, DRONE_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

def extract_row_col(filename):
    match = re.search(r"row(\d+)_col(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def imread_gray(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def connected_components_centroids_by_class(mask):
    classes = np.unique(mask)
    out = {}
    for c in classes:
        bin_img = (mask == c).astype(np.uint8)
        if bin_img.sum() == 0:
            out[int(c)] = np.zeros((0,3))
            continue
        num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_img, 8)
        rows = []
        for i in range(1, num):
            cx, cy = cents[i]
            area = stats[i, cv2.CC_STAT_AREA]
            rows.append([cx, cy, area])
        out[int(c)] = np.array(rows, dtype=float)
    return out


def flatten_centroids_dict(cc_dict):
    pts = []
    labs = []
    areas = []
    for c, arr in cc_dict.items():
        if arr is None or arr.shape[0] == 0:
            continue
        pts.append(arr[:, :2])
        labs.extend([int(c)] * arr.shape[0])
        areas.extend(arr[:, 2].tolist())
    if len(pts) == 0:
        return np.zeros((0,2)), np.array([], dtype=int), np.array([])
    pts = np.vstack(pts)
    return pts, np.array(labs), np.array(areas)


def compute_weights_from_areas(areas, labels, A0=2000.0, alpha=0.6):
    if areas is None or areas.size == 0:
        return np.array([])
    w_area = np.log1p(areas) / np.log1p(A0)
    w_area = np.clip(w_area, 0.01, 1.0)
    w_class = np.array([CLASS_WEIGHTS.get(int(l), CLASS_WEIGHT_DEFAULT) for l in labels])
    weights = alpha * w_area + (1.0 - alpha) * w_class
    return weights


def build_ref_object_masks(ref_map):
    cc = connected_components_centroids_by_class(ref_map)
    masks_by_class = {}
    for cls, arr in cc.items():
        masks = []
        bin_img = (ref_map == cls).astype(np.uint8)
        if bin_img.sum() == 0:
            masks_by_class[int(cls)] = []
            continue
        num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_img, 8)
        for i in range(1, num):
            m = (labels == i).astype(np.uint8)
            masks.append(m)
        masks_by_class[int(cls)] = masks
    return masks_by_class

def voting_accumulator(p_pts, p_labels, p_weights, r_pts, r_labels, r_weights, bin_size=VOTE_BIN, topk=3):
    if p_pts.shape[0] == 0 or r_pts.shape[0] == 0:
        return None, 0.0
    votes = Counter()
    for i in range(p_pts.shape[0]):
        li = int(p_labels[i])
        idxs = np.where(r_labels == li)[0]
        if idxs.size == 0:
            continue
        for j in idxs:
            tx = int(round(r_pts[j,0] - p_pts[i,0]))
            ty = int(round(r_pts[j,1] - p_pts[i,1]))
            bx = tx // bin_size
            by = ty // bin_size
            votes[(bx,by)] += float(p_weights[i] * r_weights[j])
    if len(votes) == 0:
        return None, 0.0
    top_bins = votes.most_common(topk)
    candidates = []
    for (bx,by), _ in top_bins:
        best_local_score = -1.0
        best_local_t = None
        for dx in range(bin_size):
            for dy in range(bin_size):
                tx = bx*bin_size + dx
                ty = by*bin_size + dy
                transformed = p_pts + np.array([tx, ty])
                dmat = cdist(transformed, r_pts)
                score = 0.0
                for pi in range(transformed.shape[0]):
                    r_idx = np.argmin(dmat[pi])
                    if int(p_labels[pi]) == int(r_labels[r_idx]) and dmat[pi, r_idx] <= MAX_MATCH_DIST:
                        score += p_weights[pi] * r_weights[r_idx] * (1.0 / (1.0 + dmat[pi, r_idx]))
                if score > best_local_score:
                    best_local_score = score
                    best_local_t = (tx, ty)
        if best_local_t is not None:
            candidates.append((best_local_t, best_local_score))
    if len(candidates) == 0:
        return None, 0.0
    candidates.sort(key=lambda x: -x[1])
    tx, ty = candidates[0][0]
    return (tx, ty), float(candidates[0][1])


def ransac_translation(p_pts, p_labels, p_weights, r_pts, r_labels, r_weights, iters=RANSAC_ITERS):
    if p_pts.shape[0] == 0 or r_pts.shape[0] == 0:
        return None, 0.0
    class_idx = defaultdict(list)
    for j in range(r_pts.shape[0]):
        class_idx[int(r_labels[j])].append(j)
    best_score = -1.0
    best_t = None
    rng = np.random.default_rng(1234)
    for _ in range(iters):
        i = rng.integers(0, p_pts.shape[0])
        li = int(p_labels[i])
        cand_js = class_idx.get(li, [])
        if len(cand_js) == 0:
            continue
        j = rng.choice(cand_js)
        tx = r_pts[j,0] - p_pts[i,0]
        ty = r_pts[j,1] - p_pts[i,1]
        transformed = p_pts + np.array([tx, ty])
        dmat = cdist(transformed, r_pts)
        score = 0.0
        inliers = 0
        for pi in range(transformed.shape[0]):
            ridx = np.argmin(dmat[pi])
            if int(p_labels[pi]) == int(r_labels[ridx]) and dmat[pi, ridx] <= MAX_MATCH_DIST:
                score += p_weights[pi] * r_weights[ridx] * (1.0 / (1.0 + dmat[pi, ridx]))
                inliers += 1
        if inliers >= RANSAC_MIN_INLIERS and score > best_score:
            best_score = score
            best_t = (tx, ty)
    if best_t is None:
        return None, 0.0
    return best_t, float(best_score)

def combined_voting_ransac(
    p_pts, p_labels, p_weights,
    r_pts, r_labels, r_weights,
    patch_center,
    bin_size=VOTE_BIN,
    ransac_iters=150  #zwiekszone z 25 do 150 iteracji
):
    if p_pts.shape[0] == 0 or r_pts.shape[0] == 0:
        return None, 0.0

    votes = Counter()
    vote_contributions = {}  # szczegóły dla każdego binu
    
    for i in range(p_pts.shape[0]):
        li = int(p_labels[i])
        idxs = np.where(r_labels == li)[0]
        if idxs.size == 0:
            continue
        for j in idxs:
            tx = int(round(r_pts[j,0] - p_pts[i,0]))
            ty = int(round(r_pts[j,1] - p_pts[i,1]))
            bin_key = (tx // bin_size, ty // bin_size)
            weight = p_weights[i] * r_weights[j]
            votes[bin_key] += weight
            
            if bin_key not in vote_contributions:
                vote_contributions[bin_key] = []
            vote_contributions[bin_key].append((i, j, weight))

    if not votes:
        return None, 0.0

    #top-3 binów dla większej odporności
    top_bins = votes.most_common(3)
    (bx_best, by_best), vote_score = top_bins[0]
    
    # Normalizuj vote confidence
    total_votes = sum(votes.values())
    vote_confidence = vote_score / (total_votes + 1e-9)

    #zwiększony obszar poszukiwań dla lepszej precyzji
    refine_radius = 3  # z 2 do 3
    
    best_score = -1.0
    best_t = (
        bx_best * bin_size + bin_size // 2, 
        by_best * bin_size + bin_size // 2
    )
    best_inliers = 0

    # Identyfikacja kandydatów do RANSAC z top-3 binów
    candidate_pairs = []
    for (bx, by), _ in top_bins:
        bin_key = (bx, by)
        if bin_key in vote_contributions:
            candidate_pairs.extend(vote_contributions[bin_key])
    
    if len(candidate_pairs) == 0:
        final_pos = (
            patch_center[0] + best_t[0],
            patch_center[1] + best_t[1]
        )
        return final_pos, float(vote_confidence)

    # Probabilistyczne próbkowanie z wagami
    weights_array = np.array([w for _, _, w in candidate_pairs])
    weights_array = weights_array / (weights_array.sum() + 1e-9)
    
    rng = np.random.default_rng(42)
    
    # RANSAC z inteligentnym próbkowaniem
    for iteration in range(ransac_iters):
        # Wybierz parę z prawdopodobieństwem proporcjonalnym do wagi
        idx = rng.choice(len(candidate_pairs), p=weights_array)
        i, j, _ = candidate_pairs[idx]
        
        # Propozycja translacji
        tx = r_pts[j,0] - p_pts[i,0]
        ty = r_pts[j,1] - p_pts[i,1]
        
        # Ogranicz do okolicy najlepszego binu
        if abs(tx - best_t[0]) > bin_size or abs(ty - best_t[1]) > bin_size:
            continue
        
        transformed = p_pts + np.array([tx, ty])
        dmat = cdist(transformed, r_pts)
        
        score = 0.0
        inliers = 0
        
        for pi in range(transformed.shape[0]):
            ridx = np.argmin(dmat[pi])
            dist = dmat[pi, ridx]
            
            if int(p_labels[pi]) == int(r_labels[ridx]) and dist <= MAX_MATCH_DIST:
                # Waga odwrotnie proporcjonalna do odległości
                score += p_weights[pi] * r_weights[ridx] * (1.0 / (1.0 + dist))
                inliers += 1
        
        if inliers >= RANSAC_MIN_INLIERS and score > best_score:
            best_score = score
            best_t = (tx, ty)
            best_inliers = inliers

    #refinement wokół najlepszego wyniku RANSAC
    for dx in range(-refine_radius, refine_radius + 1):
        for dy in range(-refine_radius, refine_radius + 1):
            tx = best_t[0] + dx
            ty = best_t[1] + dy

            transformed = p_pts + np.array([tx, ty])
            dmat = cdist(transformed, r_pts)
            
            score = 0.0
            inliers = 0
            
            for pi in range(p_pts.shape[0]):
                ridx = np.argmin(dmat[pi])
                dist = dmat[pi, ridx]
                
                if int(p_labels[pi]) == int(r_labels[ridx]) and dist <= MAX_MATCH_DIST:
                    score += p_weights[pi] * r_weights[ridx] * (1.0 / (1.0 + dist))
                    inliers += 1

            if inliers >= RANSAC_MIN_INLIERS and score > best_score:
                best_score = score
                best_t = (tx, ty)
                best_inliers = inliers

    # Łączymy confidence z voting (globalny konsensus) 
    # i RANSAC (lokalna jakość dopasowania)
    # RANSAC confidence - normalizuj przez maksymalny możliwy score
    max_possible_score = np.sum(p_weights) * np.max(r_weights) if r_weights.size > 0 else 1.0
    ransac_confidence = best_score / (max_possible_score + 1e-9)
    
    # Inlier ratio
    inlier_ratio = best_inliers / max(p_pts.shape[0], 1)
    
    # Fuzja: voting (40%) + RANSAC quality (40%) + inlier ratio (20%)
    final_conf = (
        0.4 * vote_confidence + 
        0.4 * ransac_confidence + 
        0.2 * inlier_ratio
    )

    final_pos = (
        patch_center[0] + best_t[0],
        patch_center[1] + best_t[1]
    )

    return final_pos, float(final_conf)

def run_method_on_patches(method_name, ref_map, ref_cc_masks, mask_files):
    ref_cc = connected_components_centroids_by_class(ref_map)
    r_pts, r_labels, r_areas = flatten_centroids_dict(ref_cc)
    r_weights = compute_weights_from_areas(r_areas, r_labels) if r_areas.size>0 else np.array([])

    results_traj = []
    results_conf = []

    for mask_path in mask_files:
        patch_mask = imread_gray(mask_path)
        if patch_mask is None:
            results_traj.append((np.nan, np.nan))
            results_conf.append(0.0)
            continue
        
        ph, pw = patch_mask.shape[:2]
        patch_center = np.array([pw//2, ph//2], dtype=float)

        p_cc = connected_components_centroids_by_class(patch_mask)
        p_pts, p_labels, p_areas = flatten_centroids_dict(p_cc)
        p_weights = compute_weights_from_areas(p_areas, p_labels) if p_areas.size>0 else np.array([])

        best_pos = None
        best_conf = 0.0

        if method_name == "voting_accumulator":
            res, score = voting_accumulator(p_pts, p_labels, p_weights, r_pts, r_labels, r_weights, bin_size=VOTE_BIN)
            if res is not None:
                tx, ty = res
                best_pos = (patch_center[0] + tx, patch_center[1] + ty)
                best_conf = score

        elif method_name == "ransac_translation":
            res, score = ransac_translation(p_pts, p_labels, p_weights, r_pts, r_labels, r_weights)
            if res is not None:
                tx, ty = res
                best_pos = (patch_center[0] + tx, patch_center[1] + ty)
                best_conf = score

        elif method_name == "combined_voting_ransac":
            pos, conf = combined_voting_ransac(
                p_pts, p_labels, p_weights,
                r_pts, r_labels, r_weights,
                patch_center
            )
            if pos is not None:
                best_pos = pos
                best_conf = conf

        # Fallback
        if best_pos is None:
            row, col = extract_row_col(mask_path)
            best_pos = (col + pw//2, row + ph//2)
            best_conf = 0.0

        results_traj.append(tuple(best_pos))
        results_conf.append(float(best_conf))

    return np.array(results_traj, dtype=float), np.array(results_conf, dtype=float)

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return intersection / union if union > 0 else 0.0


def extract_ref_patch(ref_map, top_left, patch_shape):
    x, y = int(round(top_left[0])), int(round(top_left[1]))
    h, w = patch_shape
    ref_h, ref_w = ref_map.shape[:2]
    
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(ref_w, x + w)
    y2 = min(ref_h, y + h)
    
    if x1 >= x2 or y1 >= y2:
        return None, 0.0
    
    ref_patch = ref_map[y1:y2, x1:x2]
    covered_area = (y2 - y1) * (x2 - x1)
    full_area = h * w
    coverage = covered_area / full_area
    
    return ref_patch, coverage


def compute_semantic_iou_patch_vs_ref(patch_mask, ref_patch):
    if ref_patch is None:
        return 0.0
    h, w = ref_patch.shape
    patch_crop = patch_mask[:h, :w]
    intersection = np.sum(patch_crop == ref_patch)
    union = patch_crop.size
    return intersection / union if union > 0 else 0.0


def compute_localization_error_top_left(pred_center, gt_top_left, patch_shape):
    ph, pw = patch_shape
    pred_top_left = np.array([
        pred_center[0] - pw / 2.0,
        pred_center[1] - ph / 2.0
    ])
    return float(np.linalg.norm(pred_top_left - np.array(gt_top_left)))

def plot_trajectory_with_gt(ref_map, traj, conf, gt_positions, method, out_dir):
    plt.figure(figsize=(12, 10))
    plt.imshow(ref_map, cmap='magma', alpha=0.7)
    
    if traj.shape[0] > 0 and gt_positions.shape[0] > 0:
        # Plot GT positions (zielone krzyżyki)
        gt_xs = gt_positions[:, 0]
        gt_ys = gt_positions[:, 1]
        plt.scatter(gt_xs, gt_ys, s=100, marker='x', c='lime', linewidths=2, 
                   label='Ground Truth', zorder=5, edgecolors='black')
        
        pred_xs = traj[:, 0]
        pred_ys = traj[:, 1]
        
        for i, (px, py, c) in enumerate(zip(pred_xs, pred_ys, conf)):
            if not np.isfinite(px):
                continue
            color = 'lime' if c > 0.6 else ('yellow' if c > 0.3 else 'red')
            plt.scatter([px], [py], s=80, c=color, edgecolors='k', zorder=4)
        
        valid = np.isfinite(pred_xs) & np.isfinite(pred_ys)
        plt.plot(pred_xs[valid], pred_ys[valid], '-', linewidth=2, 
                color='cyan', alpha=0.6, label='Predykcja', zorder=3)
        
        plt.plot(gt_xs, gt_ys, '--', linewidth=2, 
                color='lime', alpha=0.6, label='GT trajectory', zorder=3)
        
        for i in range(min(len(pred_xs), len(gt_xs))):
            if np.isfinite(pred_xs[i]):
                plt.plot([gt_xs[i], pred_xs[i]], [gt_ys[i], pred_ys[i]], 
                        'r-', alpha=0.3, linewidth=1, zorder=2)
    
    method_name_pl = METHODS_POLISH_NAMES.get(method, method)
    plt.title(f"Trajektoria vs Ground Truth – {method_name_pl}", fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    
    vispath = os.path.join(out_dir, f"traj_with_gt_{method}.png")
    plt.savefig(vispath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory with GT: {vispath}")


def plot_error_vs_frame(errors, method, out_dir):
    plt.figure(figsize=(10,4))
    plt.plot(errors, '-o', markersize=4)
    plt.xlabel("Frame index")
    plt.ylabel("Localization error [px]")
    plt.title(f"Localization error per frame – {METHODS_POLISH_NAMES.get(method, method)}")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, f"error_vs_frame_{method}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_iou_vs_frame(ious, method, out_dir):
    plt.figure(figsize=(10,4))
    plt.plot(ious, '-o', markersize=4, color='green')
    plt.xlabel("Frame index")
    plt.ylabel("Semantic IoU")
    plt.title(f"Semantic IoU per frame – {METHODS_POLISH_NAMES.get(method, method)}")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, f"iou_vs_frame_{method}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_error_hist(errors, method, out_dir):
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=30)
    plt.xlabel("Localization error [px]")
    plt.ylabel("Count")
    plt.title(f"Localization error histogram – {METHODS_POLISH_NAMES.get(method, method)}")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, f"error_hist_{method}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_error_heatmap(ref_map, traj, errors, gt_positions, method, out_dir):
    plt.figure(figsize=(12,10))
    plt.imshow(ref_map, cmap='gray', alpha=0.7)
    
    # GT positions
    plt.scatter(gt_positions[:,0], gt_positions[:,1], 
               marker='x', s=100, c='lime', linewidths=2, 
               label='Ground Truth', zorder=5, edgecolors='black')
    
    # Predicted positions colored by error
    sc = plt.scatter(
        traj[:,0], traj[:,1],
        c=errors,
        cmap='jet',
        s=80,
        edgecolors='k',
        zorder=4
    )
    plt.colorbar(sc, label="Localization error [px]")
    plt.title(f"Spatial distribution of localization error – {METHODS_POLISH_NAMES.get(method, method)}")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    
    path = os.path.join(out_dir, f"error_heatmap_{method}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    ref_map = imread_gray(REF_MAP_PATH)
    print(f"Reference map shape: {ref_map.shape}")
    if ref_map is None:
        raise FileNotFoundError(f"Reference map not found: {REF_MAP_PATH}")
    
    mask_files = sorted(glob.glob(PATCH_GLOB), key=lambda f: extract_row_col(f))
    print(f"Found {len(mask_files)} mask files")
    
    ref_cc_masks = build_ref_object_masks(ref_map)

    gt_positions = []
    for mask_path in mask_files:
        row, col = extract_row_col(mask_path)
        # Load patch to get its size
        patch_mask = imread_gray(mask_path)
        if patch_mask is not None:
            ph, pw = patch_mask.shape[:2]
            # GT center = top-left + half size
            gt_center_x = col + pw / 2.0
            gt_center_y = row + ph / 2.0
            gt_positions.append((gt_center_x, gt_center_y))
        else:
            gt_positions.append((col, row))  # Fallback
    gt_positions = np.array(gt_positions, dtype=float)
    print(f"Built GT positions (centers) for {len(gt_positions)} patches")

    summary = {}
    
    for method in METHODS_TO_RUN:
        print(f"\n{'='*80}")
        print(f"Running method: {method}")
        print(f"{'='*80}")
        
        t0 = time.time()
        traj, conf = run_method_on_patches(method, ref_map, ref_cc_masks, mask_files)
        t1 = time.time()
        print(f"Method {method} finished in {t1-t0:.2f}s")

        # Save trajectory
        outp = os.path.join(OUT_DIR, f"trajectory_{method}.npy")
        np.save(outp, {"trajectory": traj, "confidences": conf, "masks": mask_files})
        
        valid = np.isfinite(traj[:,0])
        avg_conf = float(np.mean(conf[valid])) if valid.any() else 0.0
        summary[method] = {
            "time_s": t1-t0, 
            "avg_conf": avg_conf, 
            "valid": int(np.sum(valid))
        }
        print(f"Saved {outp}; avg_conf={avg_conf:.3f}; valid={np.sum(valid)} / {len(mask_files)}")

        plot_trajectory_with_gt(ref_map, traj, conf, gt_positions, method, OUT_DIR)

        loc_errors = []
        ious = []
        coverages = []

        for (pred_center, gt_center, mask_path) in zip(traj, gt_positions, mask_files):
            if not np.isfinite(pred_center[0]):
                continue

            patch_mask = imread_gray(mask_path)
            ph, pw = patch_mask.shape[:2]

            err = float(np.linalg.norm(np.array(pred_center) - np.array(gt_center)))
            loc_errors.append(err)

            # IoU vs reference fragment
            pred_top_left = (
                pred_center[0] - pw / 2.0,
                pred_center[1] - ph / 2.0
            )

            ref_patch, coverage = extract_ref_patch(ref_map, pred_top_left, (ph, pw))
            iou = compute_semantic_iou_patch_vs_ref(patch_mask, ref_patch)

            ious.append(iou)
            coverages.append(coverage)

        mean_loc_error = float(np.mean(loc_errors)) if loc_errors else None
        mean_iou = float(np.mean(ious)) if ious else None
        mean_coverage = float(np.mean(coverages)) if coverages else None

        plot_error_vs_frame(loc_errors, method, OUT_DIR)
        plot_iou_vs_frame(ious, method, OUT_DIR)
        plot_error_hist(loc_errors, method, OUT_DIR)
        plot_error_heatmap(ref_map, traj[:len(loc_errors)], loc_errors, 
                          gt_positions[:len(loc_errors)], method, OUT_DIR)

        print(f"\nMean localization error (center-to-center): {mean_loc_error:.2f} px")
        print(f"Mean semantic IoU (patch vs ref): {mean_iou:.3f}")
        print(f"Mean reference coverage: {mean_coverage:.3f}")

        summary[method]["mean_loc_error_px"] = mean_loc_error
        summary[method]["mean_iou_patch_vs_ref"] = mean_iou
        summary[method]["mean_ref_coverage"] = mean_coverage

    summary_path = os.path.join(OUT_DIR, "methods_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("Summary written to", summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()