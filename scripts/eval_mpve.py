"""
MPVE (Mean Per-Vertex Error) batch evaluation script.

Computes per-vertex L2 error between predicted and reference garment meshes,
aggregated by garment type.

Usage:
    # Compare two rollout directories (e.g. our method vs HOOD)
    python scripts/eval_mpve.py \
        --pred_dir <path/to/our_rollouts> \
        --ref_dir <path/to/hood_rollouts> \
        --output results.csv

    # Compare a single pair of rollout files
    python scripts/eval_mpve.py \
        --pred_file <path/to/pred.pkl> \
        --ref_file <path/to/ref.pkl>

Rollout .pkl format:
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
from tqdm import tqdm


def compute_mpve(pred_verts, ref_verts):
    """
    Compute Mean Per-Vertex Error between two vertex arrays.

    Args:
        pred_verts: [V, 3] or [N, V, 3] predicted vertices
        ref_verts:  [V, 3] or [N, V, 3] reference vertices

    Returns:
        mpve: float, mean L2 distance over all vertices (and frames)
        per_frame_mpve: [N] array of per-frame MPVE (or [1])
    """
    assert pred_verts.shape == ref_verts.shape, \
        f'Shape mismatch: {pred_verts.shape} vs {ref_verts.shape}'

    if pred_verts.ndim == 2:
        errors = np.linalg.norm(pred_verts - ref_verts, axis=1)
        mpve = float(np.mean(errors))
        per_frame_mpve = np.array([mpve])
    else:
        N = pred_verts.shape[0]
        per_frame_mpve = np.zeros(N)
        for t in range(N):
            errors = np.linalg.norm(pred_verts[t] - ref_verts[t], axis=1)
            per_frame_mpve[t] = float(np.mean(errors))
        mpve = float(np.mean(per_frame_mpve))
    return mpve, per_frame_mpve


def compute_mpve_sequence(pred_path, ref_path, skip_first=2):
    """
    Compute MPVE between a predicted rollout and a reference rollout.

    Args:
        pred_path: path to predicted rollout .pkl
        ref_path: path to reference rollout .pkl
        skip_first: number of initial frames to skip

    Returns:
        dict with mpve_mm, per_frame_mpve, n_frames, vertex_count
    """
    with open(pred_path, 'rb') as f:
        pred_data = pickle.load(f)
    with open(ref_path, 'rb') as f:
        ref_data = pickle.load(f)

    pred = pred_data['pred']
    ref = ref_data['pred']

    N_pred = pred.shape[0]
    N_ref = ref.shape[0]
    N = min(N_pred, N_ref)

    frames = range(skip_first, N)
    pred_frames = pred[skip_first:N]
    ref_frames = ref[skip_first:N]

    if pred_frames.shape[0] == 0:
        return {'mpve_mm': 0.0, 'per_frame_mpve': np.array([]),
                'n_frames': 0, 'vertex_count': pred.shape[1]}

    mpve, per_frame = compute_mpve(pred_frames, ref_frames)
    mpve_mm = mpve * 1000.0  # convert meters to mm

    return {
        'mpve_mm': mpve_mm,
        'per_frame_mpve': per_frame * 1000.0,
        'n_frames': len(frames),
        'vertex_count': pred.shape[1],
    }


def find_rollout_files(rollouts_dir):
    """Find all rollout .pkl files. Layout: <rollouts_dir>/<garment>/<seq>.pkl"""
    rollouts_dir = Path(rollouts_dir)
    files = list(rollouts_dir.glob('*/*.pkl'))
    grouped = defaultdict(list)
    for fpath in files:
        garment = fpath.parent.name
        grouped[garment].append(fpath)
    return dict(grouped)


def build_seq_map(grouped):
    """Build dict: seq_name -> (garment, full_path)."""
    seq_map = {}
    for garment, paths in grouped.items():
        for p in paths:
            seq_map[p.stem] = (garment, p)
    return seq_map


def main():
    parser = argparse.ArgumentParser(
        description='MPVE batch evaluation')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Directory of predicted rollout .pkl files')
    parser.add_argument('--ref_dir', type=str, default=None,
                        help='Directory of reference rollout .pkl files')
    parser.add_argument('--pred_file', type=str, default=None,
                        help='Single predicted rollout .pkl file')
    parser.add_argument('--ref_file', type=str, default=None,
                        help='Single reference rollout .pkl file')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional CSV output path')
    parser.add_argument('--skip_first', type=int, default=2,
                        help='Frames to skip at start (default: 2)')
    args = parser.parse_args()

    # Single-file mode
    if args.pred_file and args.ref_file:
        result = compute_mpve_sequence(
            args.pred_file, args.ref_file, skip_first=args.skip_first)
        print(f'MPVE: {result["mpve_mm"]:.3f} mm')
        print(f'Frames: {result["n_frames"]}, Vertices: {result["vertex_count"]}')
        if args.output:
            with open(args.output, 'w') as f:
                f.write('pred_file,ref_file,mpve_mm,n_frames,vertex_count\n')
                f.write(f'{args.pred_file},{args.ref_file},'
                        f'{result["mpve_mm"]:.4f},{result["n_frames"]},'
                        f'{result["vertex_count"]}\n')
        return

    # Directory mode
    if not args.pred_dir or not args.ref_dir:
        print('Error: specify either (--pred_file + --ref_file) or '
              '(--pred_dir + --ref_dir)')
        sys.exit(1)

    pred_grouped = find_rollout_files(args.pred_dir)
    ref_grouped = find_rollout_files(args.ref_dir)

    if not pred_grouped:
        print(f'Error: no .pkl files found under {args.pred_dir}')
        sys.exit(1)
    if not ref_grouped:
        print(f'Error: no .pkl files found under {args.ref_dir}')
        sys.exit(1)

    pred_map = build_seq_map(pred_grouped)
    ref_map = build_seq_map(ref_grouped)

    # Match sequences by name
    common_seqs = sorted(set(pred_map.keys()) & set(ref_map.keys()))
    if not common_seqs:
        print('Error: no matching sequence names between pred and ref')
        print(f'  Pred sequences: {sorted(pred_map.keys())[:5]}...')
        print(f'  Ref sequences: {sorted(ref_map.keys())[:5]}...')
        sys.exit(1)

    print(f'Matching sequences: {len(common_seqs)}')
    print()

    # Evaluate
    results = {}
    garment_summary = defaultdict(list)

    for seq_name in tqdm(common_seqs, desc='Computing MPVE'):
        pred_garment, pred_path = pred_map[seq_name]
        _, ref_path = ref_map[seq_name]

        result = compute_mpve_sequence(pred_path, ref_path, args.skip_first)
        results[seq_name] = result
        results[seq_name]['garment'] = pred_garment
        garment_summary[pred_garment].append(result['mpve_mm'])

    # Print summary
    print()
    print('=' * 55)
    print(f'{"Garment":<20} {"Seqs":>6} {"MPVE (mm)":>14} {"Std (mm)":>12}')
    print('-' * 55)

    all_mpves = []
    for garment in sorted(garment_summary.keys()):
        mpves = garment_summary[garment]
        mean_mpve = np.mean(mpves)
        std_mpve = np.std(mpves)
        print(f'{garment:<20} {len(mpves):>6} {mean_mpve:>14.2f} {std_mpve:>12.2f}')
        all_mpves.extend(mpves)

    if all_mpves:
        print('-' * 55)
        print(f'{"OVERALL":<20} {len(all_mpves):>6} '
              f'{np.mean(all_mpves):>14.2f} {np.std(all_mpves):>12.2f}')
    print('=' * 55)

    # Optional CSV output
    if args.output:
        csv_path = Path(args.output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write('garment,sequence,mpve_mm,n_frames,vertex_count\n')
            for seq_name in sorted(results.keys()):
                r = results[seq_name]
                f.write(f'{r["garment"]},{seq_name},{r["mpve_mm"]:.4f},'
                        f'{r["n_frames"]},{r["vertex_count"]}\n')
        print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
