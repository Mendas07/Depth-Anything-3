"""
Batch evaluate multiple Depth Anything 3 models on a dataset with ground-truth depth.

Produces per-image CSVs and a summary CSV with mean metrics per model.

Usage (from repo root):
    python src/depth_anything_3/bench/batch_eval_models.py \
        --models da3-small da3-base \
        --dataset datasets/nyudepthv2/nyu_data/data/nyu2_train \
        --out outputs/bench_results --max-images 200 --device cuda

If a model name matches a key in `depth_anything_3.registry.MODEL_REGISTRY`, the script
will instantiate `DepthAnything3(model_name=...)` (uses local configs). Otherwise it
will call `DepthAnything3.from_pretrained(model_str)`.

"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.registry import MODEL_REGISTRY


def abs_rel(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    mask = (gt > eps) & np.isfinite(gt) & np.isfinite(pred)
    if mask.sum() == 0:
        return float('nan')
    p = pred[mask]
    g = gt[mask]
    return float(np.mean(np.abs(p - g) / (g + eps)))


def sq_rel(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    mask = (gt > eps) & np.isfinite(gt) & np.isfinite(pred)
    if mask.sum() == 0:
        return float('nan')
    p = pred[mask]
    g = gt[mask]
    return float(np.mean(((p - g) ** 2) / (g + eps)))


def rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    mask = (gt > eps) & np.isfinite(gt) & np.isfinite(pred)
    if mask.sum() == 0:
        return float('nan')
    p = pred[mask]
    g = gt[mask]
    return float(np.sqrt(np.mean((p - g) ** 2)))


def rmse_log(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    mask = (gt > eps) & np.isfinite(gt) & np.isfinite(pred)
    if mask.sum() == 0:
        return float('nan')
    p = pred[mask]
    g = gt[mask]
    return float(np.sqrt(np.mean((np.log(p + eps) - np.log(g + eps)) ** 2)))


def mae(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    mask = (gt > eps) & np.isfinite(gt) & np.isfinite(pred)
    if mask.sum() == 0:
        return float('nan')
    p = pred[mask]
    g = gt[mask]
    return float(np.mean(np.abs(p - g)))


def delta_acc(pred: np.ndarray, gt: np.ndarray, thr: float = 1.25, eps: float = 1e-6) -> float:
    mask = (gt > eps) & np.isfinite(gt) & np.isfinite(pred)
    if mask.sum() == 0:
        return float('nan')
    p = pred[mask]
    g = gt[mask]
    ratio = np.maximum(p / (g + eps), g / (p + eps))
    return float((ratio < thr).mean())


def extract_depth_from_prediction(prediction) -> np.ndarray:
    # prediction.depth may be torch tensor or numpy; ensure (H, W)
    depth = getattr(prediction, 'depth', None)
    if depth is None:
        raise ValueError('Prediction has no depth attribute')
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()
    # If batch dimension present
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f'Depth has unexpected shape: {depth.shape}')
    return depth.astype(np.float32)


def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for scene in sorted(root.iterdir()):
        if not scene.is_dir():
            continue
        for jpg_path in sorted(scene.glob('*.jpg')):
            gt_path = scene / f'{jpg_path.stem}.png'
            if gt_path.exists():
                pairs.append((jpg_path, gt_path))
    return pairs


def evaluate_model_on_pairs(
    model_identifier: str,
    pairs: List[Tuple[Path, Path]],
    out_dir: Path,
    device: str = 'cuda',
    max_images: int | None = None,
) -> dict:
    """Run inference for a model over pairs and compute metrics; returns mean metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = out_dir / 'per_image.csv'

    # Load model: prefer local config name if present in MODEL_REGISTRY
    if model_identifier in MODEL_REGISTRY:
        model = DepthAnything3(model_name=model_identifier)
    else:
        # assume huggingface repo id or local path
        model = DepthAnything3.from_pretrained(model_identifier)

    model = model.to(device)
    model.eval()

    results = []
    count = 0

    with torch.no_grad():
        for idx, (rgb_path, gt_path) in enumerate(pairs):
            if max_images is not None and count >= max_images:
                break
            try:
                img = cv2.imread(str(rgb_path))
                if img is None:
                    print(f'WARN: could not read {rgb_path}'); continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
                if gt is None:
                    print(f'WARN: could not read GT {gt_path}'); continue
                gt = gt.astype(np.float32) / 1000.0

                prediction = model.inference(image=[img], export_dir=None)
                pred = extract_depth_from_prediction(prediction)
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

                m_absrel = abs_rel(pred, gt)
                m_sqrel = sq_rel(pred, gt)
                m_rmse = rmse(pred, gt)
                m_rmse_log = rmse_log(pred, gt)
                m_mae = mae(pred, gt)
                m_a1 = delta_acc(pred, gt, thr=1.25)
                m_a2 = delta_acc(pred, gt, thr=1.25 ** 2)
                m_a3 = delta_acc(pred, gt, thr=1.25 ** 3)

                results.append(
                    (
                        rgb_path.name,
                        m_absrel,
                        m_sqrel,
                        m_rmse,
                        m_rmse_log,
                        m_mae,
                        m_a1,
                        m_a2,
                        m_a3,
                    )
                )

                # Save raw depth and vis
                np.save(out_dir / f'{rgb_path.stem}.npy', pred)
                pred_norm = (pred - np.nanmin(pred)) / (np.nanmax(pred) - np.nanmin(pred) + 1e-8)
                pred_vis = (pred_norm * 255).astype(np.uint8)
                cv2.imwrite(str(out_dir / f'{rgb_path.stem}.png'), pred_vis)

                count += 1
                print(f'[{model_identifier}] {count}/{len(pairs)} => {rgb_path.name} absrel={m_absrel:.4f}')

                # cleanup
                del img, gt, prediction, pred
                torch.cuda.empty_cache()

            except Exception as e:
                print(f'[ERROR] {model_identifier} failed on {rgb_path.name}: {e}')
                torch.cuda.empty_cache()
                continue

    # write per-image csv
    with open(per_image_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'mae', 'a1', 'a2', 'a3'])
        w.writerows(results)

    # compute mean metrics
    arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in results], dtype=float)
    mean_metrics = {}
    if arr.size == 0:
        # empty
        mean_metrics = {k: float('nan') for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'mae', 'a1', 'a2', 'a3']}
    else:
        mean_metrics = {
            'abs_rel': float(np.nanmean(arr[:, 0])),
            'sq_rel': float(np.nanmean(arr[:, 1])),
            'rmse': float(np.nanmean(arr[:, 2])),
            'rmse_log': float(np.nanmean(arr[:, 3])),
            'mae': float(np.nanmean(arr[:, 4])),
            'a1': float(np.nanmean(arr[:, 5])),
            'a2': float(np.nanmean(arr[:, 6])),
            'a3': float(np.nanmean(arr[:, 7])),
        }

    # free model
    del model
    torch.cuda.empty_cache()

    return mean_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=False, default=['da3-small', 'da3-base'], help='List of model identifiers (local config keys or HF repo ids)')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset root containing scenes with .jpg and matching .png ground-truths')
    parser.add_argument('--out', type=str, default='outputs/bench_results', help='Output root directory')
    parser.add_argument('--max-images', type=int, default=200, help='Max images per model (use 0 or -1 for unlimited)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if args.dataset is None:
        # Default NYU path used in repository
        dataset_root = project_root / 'datasets' / 'nyudepthv2' / 'nyu_data' / 'data' / 'nyu2_train'
    else:
        dataset_root = Path(args.dataset)

    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset root not found: {dataset_root}')

    pairs = collect_pairs(dataset_root)
    if len(pairs) == 0:
        raise RuntimeError(f'No RGB/GT pairs found under {dataset_root}')

    max_images = None if args.max_images is None or args.max_images <= 0 else args.max_images

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_id in args.models:
        print(f'\n=== Evaluating model: {model_id} ===')
        model_out = out_root / model_id.replace('/', '_')
        model_out.mkdir(parents=True, exist_ok=True)
        mean_metrics = evaluate_model_on_pairs(model_id, pairs, model_out, device=args.device, max_images=max_images)
        row = {'model': model_id}
        row.update(mean_metrics)
        summary_rows.append(row)

    # write summary csv
    summary_csv = out_root / 'summary.csv'
    keys = ['model', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'mae', 'a1', 'a2', 'a3']
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f'\nDone. Summary saved to: {summary_csv}')


if __name__ == '__main__':
    main()
