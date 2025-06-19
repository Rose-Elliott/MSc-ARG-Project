import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import pdist, squareform

# Load data
arg_data = pd.read_excel("ARG_MGE_.xlsx")
meta = pd.read_excel("cleanmeta_noLM.xlsx")

# Merge and pivot
merged = arg_data.merge(meta[['sample_id', 'Site', 'group']], on='sample_id')
pivot = merged.pivot_table(index='sample_id', columns='ARG', values='RNum_Gi',
                           aggfunc='sum', fill_value=0)
pivot = pivot.merge(meta[['sample_id', 'Site', 'group']], left_index=True, right_on='sample_id')

# Calculate site-group averages
site_group_avg = pivot.groupby(['Site', 'group']).mean(numeric_only=True).reset_index()
site_group_avg['sample_id'] = site_group_avg['Site'] + " | " + site_group_avg['group']

# Prepare matrix
arg_matrix = site_group_avg.drop(columns=['Site', 'group'])
arg_matrix = arg_matrix.set_index('sample_id')

# Distance + PCoA
bc_dist_array = pdist(arg_matrix.values, metric='braycurtis')
bc_dist = DistanceMatrix(squareform(bc_dist_array), ids=arg_matrix.index)
pcoa_res = pcoa(bc_dist)

coords = pcoa_res.samples.iloc[:, :2]
coords.columns = ['PCoA1', 'PCoA2']
coords = coords.reset_index().rename(columns={'index': 'sample_id'})
coords[['Site', 'group']] = coords['sample_id'].str.split(" \| ", expand=True)


# Ellipse function
def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                      facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)


# Plot setup
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

fig, ax = plt.subplots(figsize=(7, 6))

# Background + grid
ax.set_facecolor('#eaeaea')
ax.grid(True, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
ax.set_axisbelow(True)

# Colors by treatment group
palette = {
    'influent': '#1b7837',
    'final effluent': '#762a83'
}

# Plot points and ellipses by group
for group, subset in coords.groupby('group'):
    color = palette.get(group, '#777777')

    ax.scatter(subset['PCoA1'], subset['PCoA2'],
               color=color, edgecolor='black',
               s=90, alpha=0.9, label=group.capitalize(), zorder=3)

    draw_confidence_ellipse(subset['PCoA1'], subset['PCoA2'], ax,
                            n_std=2, edgecolor=color, linewidth=2,
                            facecolor=color, alpha=0.2, zorder=2)

# Axis labels
variance_explained = pcoa_res.proportion_explained * 100
ax.set_xlabel(f'PCoA1 ({variance_explained[0]:.1f}%)', fontsize=14, fontweight='bold')
ax.set_ylabel(f'PCoA2 ({variance_explained[1]:.1f}%)', fontsize=14, fontweight='bold')

# Legend
ax.legend(title='Sample Type', fontsize=11, title_fontsize=12,
          frameon=True, loc='lower right', fancybox=True, shadow=False)

# Final layout
plt.tight_layout()
plt.show()

# Summary
print(f"\nSummary:")
print(f"Total variance explained by first two axes: {variance_explained[:2].sum():.1f}%")
print(f"Sample sizes (number of sites per group): {coords['group'].value_counts().to_dict()}")
