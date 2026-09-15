"""Conditional probability heatmaps from saved Wolf telemetry."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
args = p.parse_args()
out = args.directory
meta = json.loads((out / 'telemetry_metadata.json').read_text())
a = np.load(out / 'telemetry.npz')['samples']
a = a[a[:, 4] > .5]
speeds = np.unique(a[:, 3])

def short(s):
    for old, new in [('front_left','FL'),('front_right','FR'),('back_left','BL'),('back_right','BR'),('Foot_',''),('_joint',''),('_link','')]:
        s=s.replace(old,new)
    return s

metrics = [('foot_force', np.linalg.norm(a[:,5:17].reshape(-1,4,3), axis=2), meta['feet'], 'Force magnitude [N] (contact >1 N)'),
           ('joint_position', a[:,17:31]*180/np.pi, meta['joints'], 'Position [deg], URDF sign'),
           ('joint_velocity', a[:,31:45], meta['joints'], 'Velocity [rad/s], URDF sign'),
           ('joint_torque', a[:,45:59], meta['joints'], 'Applied torque [Nm], URDF sign')]
for key, values, names, label in metrics:
    foot = key == 'foot_force'
    bins = np.linspace(0 if foot else values.min(), values.max(), 81)
    fig, axes = plt.subplots(1 if foot else 4, 4, figsize=(14,5) if foot else (14,13), squeeze=False, constrained_layout=True)
    matrices=[]
    for j in range(values.shape[1]):
        columns=[]
        for speed in speeds:
            v=values[a[:,3]==speed,j]
            if foot: v=v[v>1]
            h,_=np.histogram(v,bins=bins)
            assert h.sum()==len(v)
            columns.append(h / max(1,h.sum()))
        matrices.append(np.array(columns).T)
    vmax=max(m.max() for m in matrices)
    norm=LogNorm(vmin=1e-4, vmax=max(vmax,1.01e-4))
    # Homologous joint rows; leg columns FL/FR/BL/BR, absent front AFE empty.
    for ax in axes.flat: ax.axis('off')
    for j,name in enumerate(names):
        if foot: ax=axes[0,j]
        else:
            kind=name.split('_')[0];leg=short(name).split('_')[1]
            ax=axes[['HAA','HFE','KFE','AFE'].index(kind),['FL','FR','BL','BR'].index(leg)]
        ax.axis('on')
        im=ax.pcolormesh(np.arange(len(speeds)+1)-.5,bins,np.ma.masked_equal(matrices[j],0),norm=norm,cmap='magma',shading='flat')
        ax.set_xticks(range(len(speeds)),[f'{s:g}' for s in speeds])
        ax.set(title=short(name),xlabel='Command speed [m/s]',ylabel=label)
    fig.colorbar(im,ax=list(axes.flat),label='Probability per bin within each speed (log color)',shrink=.8)
    fig.suptitle(f'{meta.get("terrain","terrain").title()} | {key.replace("_"," ")} | exclude first 0.5 s\nMeasured speeds only; equal-width bins, shared axes/color within figure')
    fig.savefig(out/f'{key}_heatmap.png',dpi=150)
    plt.close(fig)
print('HEATMAP_PASS')
