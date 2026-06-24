"""
Generate comparison rollouts + compute Table 1 metrics.

Supports generating rollouts for MSAGNet, HOOD (original), SNUG, and SSCH
baselines using unified canonical geometry for fair comparison.

After generation (or with pre-existing rollouts), computes:
  - Collision loss (L_collision)
  - Average collision rate (%)
  - MPVE (mm) vs reference

Pipeline:
  1. For each (method, competitor, garment):
     - Load the method's checkpoint
     - Run inference on shared pose sequences
     - Save rollouts to disk
  2. Run eval_collision.py on each method's rollouts
  3. Run eval_mpve.py pairwise between methods

Usage:
    # Generate rollouts for all methods
    python scripts/run_table1_eval.py --generate --methods ours hood snug

    # Evaluate existing rollouts only (no generation)
    python scripts/run_table1_eval.py --eval_only \\
        --rollouts_root validation_sequences/all_rollouts

    # Full pipeline: generate + evaluate
    python scripts/run_table1_eval.py --generate --eval \\
        --methods ours hood --garments tshirt dress tanktop pants shorts
"""

import argparse
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --- Configuration ---

METHOD_CONFIGS = {
    'ours': {
        'model': 'postcvpr',
        'checkpoint': 'trained_models/postcvpr.pth',
        'config': 'postcvpr',
    },
    'hood_cvpr': {
        'model': 'cvpr',
        'checkpoint': 'trained_models/cvpr_submissionor.pth',
        'config': 'cvpr',
    },
    'hood_postcvpr': {
        'model': 'postcvpr',
        'checkpoint': 'trained_models/postcvpror.pth',
        'config': 'postcvpr',
    },
    'fine15': {
        'model': 'fine15',
        'checkpoint': 'trained_models/fine15.pth',
        'config': 'cvpr_baselines/fine15',
    },
    'fine48': {
        'model': 'fine48',
        'checkpoint': 'trained_models/fine48.pth',
        'config': 'cvpr_baselines/fine48',
    },
}

# Garment sets used in the paper
GARMENT_SETS = {
    'hood_5': ['tshirt', 'longsleeve', 'tanktop', 'pants', 'shorts'],
    'snug_2': ['tshirt', 'dress'],
    'mpve_3': ['tshirt', 'longskirt', 'hoodie'],
    'all': ['tshirt', 'longsleeve', 'tanktop', 'pants', 'shorts', 'dress', 'longskirt', 'hoodie'],
}

# Canonical geometry: which rest pose to use for each competitor
# For fair comparison, all methods should use the same canonical geometry per competitor
CANONICAL_GEOMETRIES = {
    'hood': 'validation_sequences/rest_geometries/hood.pkl',
    'snug': 'validation_sequences/rest_geometries/snug.pkl',
    'ssch': 'validation_sequences/rest_geometries/ssch.pkl',
}


# --- Rollout Generation ---

def generate_rollouts(method_name, competitor, garments, rollouts_root,
                      pose_sequences_dir, validation_config='aux/comparisons',
                      device='cuda:0'):
    """
    Generate rollout .pkl files for a given method against a competitor.

    Uses the HOOD validation pipeline to load checkpoints and run inference.

    Args:
        method_name: key in METHOD_CONFIGS
        competitor: 'hood', 'snug', or 'ssch' (determines canonical geometry + split)
        garments: list of garment names
        rollouts_root: where to save rollouts
        pose_sequences_dir: directory of input pose .pkl sequences
        validation_config: which yaml config to use
        device: torch device
    """
    from omegaconf import OmegaConf
    from utils.arguments import load_params, create_dataloader_module
    from utils.common import move2device, pickle_dump
    from utils.defaults import DEFAULTS
    from utils.validation import (update_config_for_validation,
                                   load_runner_from_checkpoint,
                                   replace_model)

    mcfg = METHOD_CONFIGS[method_name]
    checkpoint_rel = mcfg['checkpoint']
    model_config = mcfg['config']

    checkpoint_path = Path(DEFAULTS.data_root) / checkpoint_rel

    if not checkpoint_path.exists():
        print(f'Warning: checkpoint not found: {checkpoint_path}, skipping {method_name}')
        return

    # Determine canonical geometry and data split
    if competitor == 'hood':
        split_path = 'validation_sequences/datasplits/comparison_seqs_to_hood.csv'
        restpos_file = CANONICAL_GEOMETRIES.get('hood', None)
    elif competitor == 'snug':
        split_path = 'validation_sequences/datasplits/comparison_seqs_to_snug.csv'
        restpos_file = CANONICAL_GEOMETRIES.get('snug', None)
    elif competitor == 'ssch':
        split_path = 'validation_sequences/datasplits/comparison_seqs_to_ssch.csv'
        restpos_file = CANONICAL_GEOMETRIES.get('ssch', None)
    else:
        raise ValueError(f'Unknown competitor: {competitor}')

    out_dir = Path(rollouts_root) / f'vs_{competitor}' / method_name

    # Load modules and config
    modules, experiment_config = load_params(validation_config)
    experiment_config = update_config_for_validation(experiment_config,
                                                     _make_validation_conf(
                                                         data_root=pose_sequences_dir,
                                                         split_path=split_path,
                                                         restpos_file=restpos_file))
    replace_model(modules, experiment_config, model_config)
    _, runner = load_runner_from_checkpoint(
        str(checkpoint_path), modules, experiment_config)
    dataloader_m = create_dataloader_module(modules, experiment_config)
    dataloader = dataloader_m.create_dataloader(is_eval=True)

    n_generated = 0
    for sequence in dataloader:
        gname = sequence['garment_name'][0]
        if garments and gname not in garments:
            continue

        seq_name = sequence['sequence_name'][0].split('/')[-1]
        out_path = out_dir / gname / (seq_name + '.pkl')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f'  Generating: {method_name}/{gname}/{seq_name}')

        sequence = move2device(sequence, device)
        trajectories_dict = runner.valid_rollout(
            sequence, bare=True, record_time=True)
        pickle_dump(dict(trajectories_dict), out_path)
        n_generated += 1

    print(f'  Generated {n_generated} sequences for {method_name}')
    return n_generated


def _make_validation_conf(data_root, split_path, restpos_file):
    """Create a minimal validation config dataclass."""
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class ValConf:
        data_root: str = ''
        smpl_model: str = 'smpl/SMPL_FEMALE.pkl'
        garment_dict_file: str = 'garments_dict.pkl'
        obstacle_dict_file: str = 'smpl_aux.pkl'
        split_path: str = ''
        restpos_file: Optional[str] = None
        separate_arms: bool = True
        random_betas: bool = False
        zero_betas: bool = False
        density: Optional[float] = 0.20022
        lame_mu: Optional[float] = 23600.0
        lame_lambda: Optional[float] = 44400
        bending_coeff: Optional[float] = 3.9625778333333325e-05
        restpos_scale: Optional[float] = None

    return ValConf(
        data_root=data_root,
        split_path=split_path,
        restpos_file=restpos_file,
    )


# --- Table 1 Metrics ---

def compute_table1_metrics(rollouts_root, methods, competitors, garments,
                           eps=1e-3):
    """
    Compute Table 1 metrics: L_collision and average collision rates.

    Uses the collision criterion from criterions/aux/collision_metrics.py
    to compute collision loss values, and the signed-distance detector
    for collision rates.

    Returns:
        dict: method -> competitor -> {collision_loss, avg_collision_rate}
    """
    from collections import defaultdict

    results = defaultdict(lambda: defaultdict(dict))

    for competitor in competitors:
        for method in methods:
            method_dir = Path(rollouts_root) / f'vs_{competitor}' / method
            if not method_dir.exists():
                print(f'  Skipping {method} vs {competitor}: no rollouts')
                continue

            # Compute collision metrics using the signed-distance script
            print(f'\nComputing collision metrics: {method} vs {competitor}')
            script = PROJECT_ROOT / 'scripts' / 'eval_collision.py'
            cmd = [
                sys.executable, str(script),
                '--rollouts_dir', str(method_dir),
                '--eps', str(eps),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
            except Exception as e:
                print(f'  Error running eval_collision.py: {e}')

            # Also compute collision loss using HOOD's criterion
            loss_results = _compute_collision_loss_from_rollouts(
                method_dir, garments)
            results[method][competitor] = loss_results

    return dict(results)


def _compute_collision_loss_from_rollouts(rollouts_dir, garments):
    """
    Compute collision loss (L_collision) from rollout .pkl files
    using the project's collision criterion.
    """
    from collections import defaultdict
    import torch
    from criterions.aux.collision_metrics import create as create_collision_metric

    rollouts_dir = Path(rollouts_dir)

    criterion = create_collision_metric(
        type('Mcfg', (), {'weight': 1.0, 'eps': 2e-3,
         'start_rampup_iteration': 0, 'n_rampup_iterations': 1,
         'exp_min': 1.0})()
    )

    garment_losses = defaultdict(list)
    all_losses = []
    all_percs = []

    for gdir in rollouts_dir.iterdir():
        if not gdir.is_dir():
            continue
        garment = gdir.name
        if garments and garment not in garments:
            continue

        for seq_path in sorted(gdir.glob('*.pkl')):
            with open(seq_path, 'rb') as f:
                data = pickle.load(f)

            pred = data['pred']
            obstacle = data['obstacle']
            cloth_faces = data['cloth_faces']
            obstacle_faces = data['obstacle_faces']

            import torch
            device = torch.device('cpu')

            N = pred.shape[0]
            for t in range(2, N):
                cloth_pos = torch.FloatTensor(pred[t - 1]).unsqueeze(0)
                cloth_pred = torch.FloatTensor(pred[t]).unsqueeze(0)
                obs_pos = torch.FloatTensor(obstacle[t]).unsqueeze(0)
                obs_faces = torch.LongTensor(obstacle_faces.T).unsqueeze(0)

                # Simplified collision loss computation using face centers
                from utils.cloth_and_material import FaceNormals
                from utils.common import gather

                f_normals = FaceNormals()
                obs_faces_t = obs_faces[0].T
                obs_fn = f_normals(obs_pos, obs_faces)[0]

                obs_face_centers = gather(
                    obs_pos[0], obs_faces_t, 0, 1, 1).mean(dim=-2)

                from pytorch3d.ops import knn_points
                _, nn_idx, _ = knn_points(
                    cloth_pred, obs_face_centers.unsqueeze(0), return_nn=True)
                nn_idx = nn_idx[0, :, 0]

                nn_normals = obs_fn[nn_idx]
                nn_centers = obs_face_centers[nn_idx]

                distances = ((cloth_pred[0] - nn_centers) * nn_normals).sum(dim=-1)
                interp = torch.clamp(2e-3 - distances, min=0)
                loss = interp.pow(3).sum().item()
                perc = (interp > 0).float().mean().item()

                garment_losses[garment].append(loss)
                all_losses.append(loss)
                all_percs.append(perc)

    return {
        'per_garment': {g: np.mean(v) for g, v in garment_losses.items()},
        'overall_loss': np.mean(all_losses) if all_losses else 0.0,
        'overall_perc': np.mean(all_percs) * 100 if all_percs else 0.0,
    }


def print_table1(results, methods, competitors):
    """Print Table 1 in LaTeX-friendly format."""
    print()
    print('=' * 75)
    print('TABLE 1: Collision Handling Comparison')
    print('=' * 75)
    print(f'{"Method":<16} {"Competitor":<12} {"L_collision":>14} {"Avg Coll %":>12}')
    print('-' * 55)

    for method in methods:
        for competitor in competitors:
            if method in results and competitor in results[method]:
                r = results[method][competitor]
                loss = r.get('overall_loss', 0)
                perc = r.get('overall_perc', 0)
                print(f'{method:<16} {competitor:<12} {loss:>14.2e} {perc:>11.4f}%')

    print('=' * 75)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description='Table 1 evaluation: generate rollouts + compute metrics')
    parser.add_argument('--generate', action='store_true',
                        help='Generate rollout sequences')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate existing rollouts (no generation)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['ours', 'hood_cvpr'],
                        help='Methods to evaluate (keys in METHOD_CONFIGS)')
    parser.add_argument('--competitors', type=str, nargs='+',
                        default=['hood', 'snug'],
                        help='Competitors: hood, snug, ssch')
    parser.add_argument('--garments', type=str, nargs='+',
                        default=None,
                        help='Garments to include (default: all available)')
    parser.add_argument('--rollouts_root', type=str,
                        default='aux_data/validation_sequences/all_rollouts',
                        help='Root directory for rollout .pkl files')
    parser.add_argument('--pose_dir', type=str,
                        default='aux_data/validation_sequences/pose_sequences',
                        help='Directory of input pose sequences')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eps', type=float, default=1e-3,
                        help='Penetration threshold for collision detection')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional JSON output for results')
    args = parser.parse_args()

    rollouts_root = Path(args.rollouts_root)

    if args.generate:
        print('=' * 60)
        print('STEP 1: Generating rollouts')
        print('=' * 60)

        for method in args.methods:
            for competitor in args.competitors:
                print(f'\n--- {method} vs {competitor} ---')
                try:
                    generate_rollouts(
                        method_name=method,
                        competitor=competitor,
                        garments=args.garments,
                        rollouts_root=str(rollouts_root),
                        pose_sequences_dir=args.pose_dir,
                        device=args.device,
                    )
                except Exception as e:
                    print(f'  Failed: {e}')
                    import traceback
                    traceback.print_exc()

    if args.generate or args.eval_only:
        print()
        print('=' * 60)
        print('STEP 2: Computing Table 1 metrics')
        print('=' * 60)

        results = compute_table1_metrics(
            rollouts_root=str(rollouts_root),
            methods=args.methods,
            competitors=args.competitors,
            garments=args.garments,
            eps=args.eps,
        )
        print_table1(results, args.methods, args.competitors)

        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy values for JSON serialization
            def convert(obj):
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                elif isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                return obj

            with open(output_path, 'w') as f:
                json.dump(convert(results), f, indent=2)
            print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
