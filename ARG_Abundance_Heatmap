import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

metadata_df = pd.read_excel('cleanmeta_noLM.xlsx')
arg_df = pd.read_excel('ARG_MGE_.xlsx')
merged_df = pd.merge(arg_df, metadata_df, on='sample_id')

filtered_df = merged_df[merged_df['group'].isin(['influent', 'final effluent'])]
agg_df = filtered_df.groupby(['ARG', 'group', 'Site'])['RNum_Gi'].sum().reset_index()
pivot_df = agg_df.pivot_table(index='ARG', columns=['group', 'Site'], values='RNum_Gi', fill_value=0)

top_args = pivot_df.sum(axis=1).sort_values(ascending=False).head(30).index
pivot_top = pivot_df.loc[top_args]

abbrev_dict = {
    "TRUNCATED_PUTATIVE_RESPONSE_REGULATOR_ARLR": "TPRR",
    "ESCHERICHIA COLI EF-TU MUTANTS CONFERRING RESISTANCE TO PULVOMYCIN": "ECEP",
    "KASUGAMYCIN_RESISTANCE_PROTEIN_KSGA": "KSGA",
    "BIFIDOBACTERIUM ADOLESCENTIS RPOB MUTANTS CONFERRING RESISTANCE TO RIFAMPICIN": "BAR-R"
}
pivot_top.index = pivot_top.index.to_series().replace(abbrev_dict)

pivot_log = np.log10(pivot_top + 1e-6)

influent = pivot_log['influent']
effluent = pivot_log['final effluent']

all_sites = set(influent.columns) | set(effluent.columns)
influent = influent.reindex(columns=sorted(all_sites))
effluent = effluent.reindex(columns=sorted(all_sites))

fig, axes = plt.subplots(1, 2, figsize=(22, 12), gridspec_kw={'width_ratios': [1, 1]})

try:
    import cmasher as cmr
    colormap = cmr.imola
    crameri_available = True
except (ImportError, AttributeError):
    crameri_available = False
    colormap = 'cividis'  #for if it doesn't work

heatmap_params = {
    'cmap': colormap,
    'linewidths': 0.1,
    'linecolor': 'white',
    'xticklabels': True
}

# Influent
sns.heatmap(influent, ax=axes[0], cbar=False, yticklabels=True, **heatmap_params)
axes[0].set_title("Influent", fontsize=14)
axes[0].tick_params(axis='x', rotation=45, labelsize=9)
axes[0].tick_params(axis='y', labelsize=9)
axes[0].set_ylabel("")

# Effluent
sns.heatmap(effluent, ax=axes[1], yticklabels=False,
           cbar_kws={'label': 'Log10(Relative Abundance + 1e-6)'}, **heatmap_params)
axes[1].set_title("Final Effluent", fontsize=14)
axes[1].tick_params(axis='x', rotation=45, labelsize=9)
axes[1].set_ylabel("")

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()
