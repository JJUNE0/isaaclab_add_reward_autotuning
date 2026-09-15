"""Summarize physics-rate Wolf telemetry without requiring Isaac Sim."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
p = args.directory
meta = json.loads((p / 'telemetry_metadata.json').read_text())
a = np.load(p / 'telemetry.npz')['samples']
assert a.shape[1] == len(meta['columns']) and np.isfinite(a).all()
forces = np.linalg.norm(a[:, 5:17].reshape(-1, 4, 3), axis=2)
metrics = {'force_N': forces, 'position_rad': a[:, 17:31],
           'velocity_rad_s': a[:, 31:45], 'applied_torque_Nm': a[:, 45:59]}
names = {'force_N': meta['feet'], **{k: meta['joints'] for k in metrics if k != 'force_N'}}
rows = []
for speed in [None, 1, 2]:
    for level in [None, *sorted(np.unique(a[:, 2]).astype(int))]:
        mask = np.ones(len(a), dtype=bool)
        if speed is not None:
            mask &= a[:, 3] == speed
        if level is not None:
            mask &= a[:, 2] == level
        for key, values in metrics.items():
            for i, name in enumerate(names[key]):
                v = values[mask, i]
                for population in (['all', 'contact_gt_1N'] if key == 'force_N' else ['all']):
                    x = v[v > 1] if population != 'all' else v
                    if not len(x):
                        continue
                    rows.append(dict(speed='all' if speed is None else speed,
                        level='all' if level is None else level, metric=key, name=name,
                        population=population, samples=len(x), mean=float(x.mean()),
                        std=float(x.std()), min=float(x.min()), p01=float(np.percentile(x, 1)),
                        p05=float(np.percentile(x, 5)), p50=float(np.percentile(x, 50)),
                        p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)),
                        p995=float(np.percentile(x, 99.5)), max=float(x.max()),
                        abs_p99=float(np.percentile(abs(x), 99)), abs_max=float(abs(x).max()),
                        contact_fraction=float((v > 1).mean()) if key == 'force_N' else ''))
with (p / 'distribution_statistics.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def short(name):
    return name.replace('front_left', 'FL').replace('front_right', 'FR').replace('back_left', 'BL').replace('back_right', 'BR').replace('_joint', '').replace('Foot_', '').replace('_link', '')

# Histogram densities integrate to one; no dropped tails or downsampling.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    for speed in [1, 2]:
        v = forces[a[:, 3] == speed, i]; v = v[v > 1]
        ax.hist(v, bins=100, density=True, histtype='step', label=f'{speed} m/s')
    ax.set(title=short(meta['feet'][i]), xlabel='Net contact force magnitude [N]', ylabel='Contact-only density')
    ax.set_yscale('log'); ax.legend(); ax.grid(alpha=.2)
fig.suptitle('Foot contact force distributions (>1 N), all levels; 200 Hz')
fig.tight_layout(); fig.savefig(p / 'foot_force_distributions.png', dpi=160); plt.close(fig)

for metric in ['position_rad', 'velocity_rad_s', 'applied_torque_Nm']:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for kind, ax in zip(['HAA', 'HFE', 'KFE', 'AFE'], axes.flat):
        ids = [j for j, name in enumerate(meta['joints']) if name.startswith(kind + '_')]
        bins = np.histogram_bin_edges(metrics[metric][:, ids].ravel(), bins=100)
        for j in ids:
            ax.hist(metrics[metric][:, j], bins=bins, density=True, histtype='step', label=short(meta['joints'][j]))
        ax.set(title=kind, xlabel=metric, ylabel='Density'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('URDF signed joint coordinates; both speeds and all levels')
    fig.tight_layout(); fig.savefig(p / f'{metric}_distributions.png', dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for speed, ax in zip([1, 2], axes):
    for i, foot in enumerate(meta['feet']):
        vals = []
        for level in sorted(np.unique(a[:, 2]).astype(int)):
            v = forces[(a[:, 3] == speed) & (a[:, 2] == level), i]; v = v[v > 1]
            vals.append(np.percentile(v, 99))
        ax.plot(sorted(np.unique(a[:, 2]).astype(int)), vals, marker='o', label=short(foot))
    ax.set(title=f'Command {speed} m/s', xlabel='Terrain level', ylabel='Contact-only force P99 [N]'); ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(p / 'force_p99_by_level.png', dpi=160); plt.close(fig)

# Episode sample accounting: exactly four physics samples per control frame.
_, counts = np.unique(a[:, :2], axis=0, return_counts=True)
assert len(counts) == meta.get("expected_episodes", 384) and np.all(counts % 4 == 0), (len(counts), counts)
summary = [r for r in rows if r['speed'] == 'all' and r['level'] == 'all']
(p / 'summary.json').write_text(json.dumps({'rows': len(a), 'episodes': len(counts), 'statistics': summary}, indent=2))
print(json.dumps({'rows': len(a), 'episodes': len(counts), 'feet_contact': [r for r in summary if r['population'] == 'contact_gt_1N']}, indent=2))

# Keep reset/transient sensitivity explicit for future reward normalization.
transient_rows = []
for cutoff in [0.0, 0.5]:
    for i, foot in enumerate(meta['feet']):
        v = forces[(a[:, 4] > cutoff) & (forces[:, i] > 1), i]
        transient_rows.append({'exclude_first_seconds': cutoff, 'foot': foot,
            'count': len(v), 'p99_N': float(np.percentile(v, 99)),
            'p995_N': float(np.percentile(v, 99.5)), 'max_N': float(v.max())})
(p / 'force_transient_sensitivity.json').write_text(json.dumps(transient_rows, indent=2))

# Speed-conditioned envelopes, excluding reset startup; retain signed position.
for metric, values in metrics.items():
    fig, axes = plt.subplots(4, 4 if metric != 'force_N' else 1,
                             figsize=(15, 12) if metric != 'force_N' else (9, 10), squeeze=False)
    for j, ax in enumerate(axes.flat):
        if j >= values.shape[1]:
            ax.axis('off'); continue
        speeds = sorted(np.unique(a[:, 3]))
        quantiles = []
        for speed in speeds:
            x = values[(a[:, 3] == speed) & (a[:, 4] > .5), j]
            if metric == 'force_N': x = x[x > 1]
            elif metric != 'position_rad': x = abs(x)
            quantiles.append(np.percentile(x, [5, 50, 95, 99.5]))
        q = np.array(quantiles)
        ax.fill_between(speeds, q[:, 0], q[:, 2], alpha=.2, label='P5–P95')
        ax.plot(speeds, q[:, 1], 'o-', label='P50')
        ax.plot(speeds, q[:, 3], 's--', label='P99.5')
        ax.set(title=short(names[metric][j]), xlabel='Command [m/s]', ylabel=metric if metric in ['force_N', 'position_rad'] else 'abs ' + metric)
        ax.grid(alpha=.2)
        if j == 0: ax.legend()
    fig.suptitle('Speed-conditioned distributions; exclude first 0.5 s; force contact-only')
    fig.tight_layout(); fig.savefig(p / f'{metric}_by_speed.png', dpi=140); plt.close(fig)
