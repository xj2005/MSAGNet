"""
Proper signed-distance collision detection + batch statistics.

Replaces the AABB-based detection in 评价指标.ipynb with correct
face-normal signed-distance penetration detection.

Usage:
    python scripts/eval_collision.py --rollouts_dir <path> [--eps 1e-3] [--output csv]

The script expects rollout .pkl files organized as:
    <rollouts_dir>/<garment_name>/<sequence_name>.pkl

Each .pkl contains:
    'pred':          np.ndarray [N_frames, V, 3]  cloth vertex positions
    'obstacle':      np.ndarray [N_frames, W, 3]  body vertex positions
    'cloth_faces':   np.ndarray [F_c, 3]          cloth face indices
    'obstacle_faces': np.ndarray [F_o, 3]         body face indices
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def compute_face_centers_and_normals(vertices, faces):
    """
    Compute face centers and outward-facing normals.

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] face indices (0-based)

    Returns:
        centers: [F, 3] face centers
        normals: [F, 3] unit face normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    centers = (v0 + v1 + v2) / 3.0
    e0 = v1 - v0
    e1 = v2 - v0
    normals = np.cross(e0, e1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals = normals / norms
    return centers, normals


def compute_body_collision(cloth_verts, body_verts, body_faces, eps=1e-3):
    """
    Compute body-garment collision rate using signed distance.

    For each cloth vertex, finds the nearest body face via face-center KNN,
    computes signed distance (positive = outside body, negative = penetrating),
    and counts vertices with signed_distance < -eps as colliding.

    Args:
        cloth_verts: [V_c, 3] cloth vertex positions
        body_verts:  [V_b, 3] body vertex positions
        body_faces:  [F_b, 3] body face indices
        eps:         penetration threshold (meters)

    Returns:
        collision_rate: float, fraction of cloth vertices penetrating the body
        collision_count: int, number of penetrating vertices
        total_vertices: int, total cloth vertices
    """
    centers, normals = compute_face_centers_and_normals(body_verts, body_faces)

    tree = cKDTree(centers)
    _, nn_idx = tree.query(cloth_verts, k=1)

    nearest_centers = centers[nn_idx]
    nearest_normals = normals[nn_idx]

    signed_dist = np.sum((cloth_verts - nearest_centers) * nearest_normals, axis=1)
    penetrating = signed_dist < -eps

    total = len(cloth_verts)
    count = int(penetrating.sum())
    rate = count / total if total > 0 else 0.0
    return rate, count, total


def compute_self_collision(cloth_verts, cloth_faces, eps=1e-3, k=6):
    """
    Compute garment self-collision rate using signed distance.

    For each cloth vertex, finds nearby cloth faces (excluding faces
    that contain the vertex), computes signed distance, and counts
    vertices with signed_distance < -eps as self-penetrating.
    Also filters out very close vertex-face pairs that are likely
    mesh-adjacent.

    Args:
        cloth_verts: [V, 3] cloth vertex positions
        cloth_faces: [F, 3] cloth face indices
        eps:         penetration threshold (meters)
        k:           number of nearby faces to check per vertex

    Returns:
        collision_rate: float
        collision_count: int
        total_vertices: int
    """
    V = len(cloth_verts)
    centers, normals = compute_face_centers_and_normals(cloth_verts, cloth_faces)

    # Build mapping: vertex_idx -> set of face indices containing that vertex
    vertex_to_faces = defaultdict(set)
    for fi, face in enumerate(cloth_faces):
        for vi in face:
            vertex_to_faces[vi].add(fi)

    tree = cKDTree(centers)
    _, nn_indices = tree.query(cloth_verts, k=k)  # [V, k]

    count = 0
    for vi in range(V):
        adjacent_faces = vertex_to_faces[vi]
        found_penetrating = False

        for nni in nn_indices[vi]:
            if nni in adjacent_faces:
                continue

            face_center = centers[nni]
            face_normal = normals[nni]

            signed_dist = np.dot(cloth_verts[vi] - face_center, face_normal)
            euclidean_dist = np.linalg.norm(cloth_verts[vi] - face_center)

            # Filter mesh-adjacent pairs by distance
            if euclidean_dist < eps:
                continue

            if signed_dist < -eps:
                found_penetrating = True
                break

        if found_penetrating:
            count += 1

    rate = count / V if V > 0 else 0.0
    return rate, count, V


def evaluate_sequence(seq_path, eps=1e-3, skip_first=2):
    """
    Evaluate collision rates for a single rollout sequence.

    Args:
        seq_path: path to .pkl rollout file
        eps: penetration threshold
        skip_first: number of initial frames to skip (warm-up)

    Returns:
        dict with avg_body_collision_rate, avg_self_collision_rate, n_frames
    """
    with open(seq_path, 'rb') as f:
        data = pickle.load(f)

    pred = data['pred']           # [N, V, 3]
    obstacle = data['obstacle']   # [N, W, 3]
    cloth_faces = data['cloth_faces']     # [F_c, 3]
    obstacle_faces = data['obstacle_faces']  # [F_o, 3]

    N = pred.shape[0]
    frames = range(skip_first, N)

    body_rates = []
    self_rates = []

    for t in frames:
        body_rate, _, _ = compute_body_collision(
            pred[t], obstacle[t], obstacle_faces, eps=eps)
        self_rate, _, _ = compute_self_collision(
            pred[t], cloth_faces, eps=eps)
        body_rates.append(body_rate)
        self_rates.append(self_rate)

    return {
        'n_frames': len(frames),
        'avg_body_collision_rate': np.mean(body_rates) if body_rates else 0.0,
        'avg_self_collision_rate': np.mean(self_rates) if self_rates else 0.0,
    }


def find_rollout_files(rollouts_dir):
    """
    Find all rollout .pkl files under rollouts_dir.
    Expected layout: <rollouts_dir>/<garment>/<seq_name>.pkl
    """
    rollouts_dir = Path(rollouts_dir)
    files = list(rollouts_dir.glob('*/*.pkl'))
    grouped = defaultdict(list)
    for fpath in files:
        garment = fpath.parent.name
        grouped[garment].append(fpath)
    return dict(grouped)


def main():
    parser = argparse.ArgumentParser(
        description='Signed-distance collision detection + batch statistics')
    parser.add_argument('--rollouts_dir', type=str, required=True,
                        help='Directory containing rollout .pkl files (organized as <garment>/<seq>.pkl)')
    parser.add_argument('--eps', type=float, default=1e-3,
                        help='Penetration threshold in meters (default: 1e-3)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional CSV output path')
    args = parser.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    if not rollouts_dir.exists():
        print(f'Error: rollouts directory not found: {rollouts_dir}')
        sys.exit(1)

    grouped = find_rollout_files(rollouts_dir)
    if not grouped:
        print(f'Error: no .pkl files found under {rollouts_dir}')
        sys.exit(1)

    print(f'Found {sum(len(v) for v in grouped.values())} sequences '
          f'across {len(grouped)} garments')
    print(f'Penetration threshold (eps): {args.eps} m')
    print()

    # Evaluate all sequences
    all_results = {}
    garment_summary = defaultdict(lambda: {
        'body_rates': [], 'self_rates': [], 'n_frames': 0})

    for garment, seq_paths in sorted(grouped.items()):
        for seq_path in tqdm(sorted(seq_paths), desc=garment):
            result = evaluate_sequence(seq_path, eps=args.eps)
            all_results[str(seq_path)] = result
            garment_summary[garment]['body_rates'].append(
                result['avg_body_collision_rate'])
            garment_summary[garment]['self_rates'].append(
                result['avg_self_collision_rate'])
            garment_summary[garment]['n_frames'] += result['n_frames']

    # Print per-garment summary
    print()
    print('=' * 75)
    print(f'{"Garment":<20} {"Seqs":>6} {"Body Coll %":>14} {"Self Coll %":>14} {"Total Coll %":>14}')
    print('-' * 75)

    all_body_rates = []
    all_self_rates = []
    total_frames = 0

    for garment in sorted(garment_summary.keys()):
        gs = garment_summary[garment]
        body_avg = np.mean(gs['body_rates']) * 100
        self_avg = np.mean(gs['self_rates']) * 100
        total_avg = body_avg + self_avg
        n_seqs = len(gs['body_rates'])
        print(f'{garment:<20} {n_seqs:>6} {body_avg:>13.4f}% {self_avg:>13.4f}% {total_avg:>13.4f}%')
        all_body_rates.extend(gs['body_rates'])
        all_self_rates.extend(gs['self_rates'])
        total_frames += gs['n_frames']

    print('-' * 75)
    overall_body = np.mean(all_body_rates) * 100
    overall_self = np.mean(all_self_rates) * 100
    overall_total = overall_body + overall_self
    print(f'{"OVERALL":<20} {sum(len(v) for v in grouped.values()):>6} '
          f'{overall_body:>13.4f}% {overall_self:>13.4f}% {overall_total:>13.4f}%')
    print('=' * 75)
    print(f'Total frames evaluated: {total_frames}')

    # Optional CSV output
    if args.output:
        csv_path = Path(args.output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write('garment,sequence,body_collision_rate,self_collision_rate\n')
            for garment, seq_paths in sorted(grouped.items()):
                for seq_path in sorted(seq_paths):
                    r = all_results[str(seq_path)]
                    f.write(f'{garment},{seq_path.stem},'
                            f'{r["avg_body_collision_rate"]:.6f},'
                            f'{r["avg_self_collision_rate"]:.6f}\n')
        print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
