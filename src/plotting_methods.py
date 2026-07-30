import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns


def annotated_umap(adata, tissue_type, sample_name, obs):
    """
    Creates a UMAP plot with annotated clusters.

    This function visualizes UMAP results from an AnnData object, coloring the points by cell type
    and annotating the clusters with their respective labels. The plot includes a title indicating
    the tissue type and sample name.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing UMAP coordinates in the obsm['X_umap'] attribute and cell type
        information in the specified observation column.
    tissue_type : str
        The type of tissue being analyzed, used in the plot title.
    sample_name : str
        The name or identifier of the sample set, used in the plot title.
    obs : str
        The name of the observation column in adata.obs that contains the cell type information.

    Returns
    -------
    None
        The function displays the UMAP plot but does not return any value.
    """
    sc.pl.umap(
        adata,
        color=[obs],
        cmap='turbo',
        title='{} {} celltypes'.format(tissue_type, sample_name),
        show=False,
    )

    ax = plt.gca()
    for cluster in adata.obs['leiden'].cat.categories:
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_coords = adata.obsm['X_umap'][cluster_mask]
        x, y = cluster_coords[:, 0].mean(), cluster_coords[:, 1].mean()
        ax.text(x, y, cluster, color='black', fontsize=10,
                weight='bold', ha='center', va='center')

    plt.show()


def RTL_boxplot(df, population_name):
    """
    Creates a boxplot with overlaid stripplot and pointplot for RTL values across different tissues.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted. Must include 'Tissue' and 'RTL' columns.
    population_name : str
        The name of the population being analyzed, used in the plot title.

    Returns
    -------
    None
        The function displays the boxplot but does not return any value.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    custom_palette = {
        'SPL': '#e41a1c',       # Explicit Red
        'LLN': '#377eb8',      # Explicit Blue
        'MLN': '#4daf4a',          # Explicit Green
        'LNG': '#984ea3'  # Explicit Purple
    }

    sns.stripplot(
        x='Tissue', y='RTL', data=df,
        hue='Tissue', legend=False,
        palette=custom_palette,
        jitter=True, size=6, edgecolor='black', linewidth=0.5, ax=ax
    )

    sns.boxplot(
        x='Tissue', y='RTL', data=df,
        hue='Tissue', palette=custom_palette, dodge=False, boxprops=dict(alpha=0.5), ax=ax
    )

    sns.pointplot(
        x='Tissue', y='RTL', data=df, color='black',
        errorbar='sd',
        linestyle='none',
        markers='_',
        native_scale=True,
        err_kws={'linewidth': 1.5},
        markersize=15,
        linewidth=2.5,
        ax=ax
    )

    pairs = [("MLN", "SPL"), ("MLN", "LLN"), ("MLN", "LNG")]

    # annotator = Annotator(ax, pairs, data=df, x='Tissue', y='RTL')
    # annotator.configure(
    #     test='Mann-Whitney',
    #     text_format='star',
    #     loc='outside',
    #     comparisons_correction='bonferroni',
    #     line_width=1.5
    # )
    # annotator.apply_and_annotate()

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.title(f'{[population_name]} RTL by Tissue', fontsize=14)
    sns.despine()

    plt.tight_layout()
    plt.show()
