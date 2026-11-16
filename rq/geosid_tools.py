"""Nearest-neighbor SID export and fixed-projection catalog diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .models.georeg import GeoReg
from .models.rqvae import RQVAE
from .flash_sid.export import build_minionerec_index, load_item_ids, validate_item_ids


def load_model(checkpoint, device='cpu'):
    # Checkpoints are trusted local files, not arbitrary downloaded pickle inputs.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    values = state.get('model_args')
    if values is None:
        values = state.get('args', {})
        values = values if isinstance(values, dict) else vars(values)
    allowed = set(inspect.signature(RQVAE).parameters)
    values = {key: value for key, value in values.items() if key in allowed}
    if 'in_dim' not in values:
        values['in_dim'] = state['state_dict']['encoder.mlp_layers.1.weight'].shape[1]
    values.update(state.get('geo_config') or {'geo_mode': 'none', 'geo_weight': 0.0})
    model = RQVAE(**values).to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    for quantizer, flag in zip(model.rq.vq_layers, state.get('vq_initted', [True] * len(model.rq.vq_layers))):
        quantizer.initted = bool(flag)
    model.eval()
    return model, state


def group_metrics(rows):
    n = len(rows)
    counts = np.asarray(list(Counter(tuple(row) for row in rows).values()), dtype=np.int64)
    return {'unique': int(len(counts)), 'duplicate_excess': float((n - len(counts)) / n),
            'collision_item_fraction': float(counts[counts > 1].sum() / n),
            'collision_groups': int((counts > 1).sum()), 'max_group': int(counts.max()),
            'same_prefix_pair_fraction': float((counts * (counts - 1)).sum() / (n * (n - 1))) if n > 1 else 0.0}


def code_metrics(codes, sizes):
    layers = []
    for level, k in enumerate(sizes):
        counts = np.bincount(codes[:, level], minlength=k)
        probabilities = counts[counts > 0] / len(codes)
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        layers.append({'layer': level + 1, 'counts': counts.tolist(),
                       'used_codes': int((counts > 0).sum()), 'usage_ratio': float((counts > 0).mean()),
                       'entropy_nats': entropy, 'normalized_entropy': entropy / np.log(k),
                       'effective_ratio': float(np.exp(entropy) / k),
                       'gini': float(np.abs(counts[:, None] - counts[None, :]).sum() / (2 * k * counts.sum()))})
    return {'layers': layers, 'prefix': {str(j): group_metrics(codes[:, :j].tolist()) for j in range(1, codes.shape[1] + 1)}}


@torch.no_grad()
def catalog_diagnostics(model, embeddings, item_ids, diagnostic_seed=314159):
    device = next(model.parameters()).device
    x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    if x.ndim != 2 or len(x) < 2 or not torch.isfinite(x).all():
        raise ValueError('Catalog must be a finite matrix with at least two items')
    ids = validate_item_ids(item_ids, len(x))
    model.eval()
    z = model.encoder(x)
    quantized, quant_loss, indices = model.rq(z, use_sk=False)
    reconstruction = model.decoder(quantized)
    codes = indices.cpu().numpy()
    final_index = build_minionerec_index(codes, ids, suffix_size=len(ids))
    raw_index = {key: tokens[:3] for key, tokens in final_index.items()}
    diag = GeoReg(seed=diagnostic_seed).to(device)
    directions = diag.sample_directions(z.shape[1], device=device)
    residual = z
    geometries = []
    for level, quantizer in enumerate(model.rq.vq_layers):
        centered = residual.double() - residual.double().mean(0)
        eig = torch.linalg.eigvalsh(centered.T @ centered / len(x)).clamp_min(0)
        probabilities = eig / eig.sum().clamp_min(1e-30)
        effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()) if eig.sum() > 0 else eig.new_zeros(())
        loss, stats = diag(residual, directions=directions, return_stats=True)
        geometries.append({'layer': level + 1, 'normalized_cf': loss.item(),
                           'residual_rms': stats['residual_rms'].item(),
                           'centered_rms': stats['scale'].item(),
                           'covariance_eigenvalues': eig.cpu().tolist(),
                           'effective_rank': effective_rank.item()})
        if level + 1 < len(model.rq.vq_layers):
            residual = residual - quantizer.get_codebook_entry(indices[:, level]).detach()
    metrics = code_metrics(codes, model.num_emb_list)
    metrics.update(items=len(ids), reconstruction_mse=(reconstruction - x).square().mean().item(),
                   quantization_loss=quant_loss.item(), geometry=geometries,
                   mean_normalized_cf=float(np.mean([g['normalized_cf'] for g in geometries])),
                   diagnostic_seed=diagnostic_seed, raw_collision=group_metrics(codes.tolist()),
                   final_collision=group_metrics(list(final_index.values())), geo_config=model.geo_config)
    return metrics, codes, raw_index, final_index


def write_outputs(output_dir, metrics, codes, raw_index, final_index):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [('metrics.json', metrics), ('raw.index.json', raw_index), ('index.json', final_index)]:
        (output_dir / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    np.save(output_dir / 'raw_codes.npy', codes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'diagnose'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--embeddings', required=True)
    alignment = parser.add_mutually_exclusive_group(required=True)
    alignment.add_argument('--item-ids')
    alignment.add_argument('--item-meta')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--diagnostic-seed', type=int, default=314159)
    args = parser.parse_args(argv)
    embeddings = np.load(args.embeddings)
    ids = load_item_ids(len(embeddings), Path(args.item_ids) if args.item_ids else None,
                        Path(args.item_meta) if args.item_meta else None)
    model, _ = load_model(args.checkpoint, args.device)
    outputs = catalog_diagnostics(model, embeddings, ids, args.diagnostic_seed)
    write_outputs(args.output_dir, *outputs)
    print(json.dumps(outputs[0]))


if __name__ == '__main__':
    main()
