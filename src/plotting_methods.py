from itertools import combinations

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from statannotations.Annotator import Annotator

custom_palette = {
    'SPL': '#e41a1c',
    'LLN': '#377eb8',
    'MLN': '#4daf4a',
    'LNG': '#984ea3'
}


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


def RTL_boxplot(df, population_name, annotate=True):
    """
    Creates a boxplot with overlaid stripplot and pointplot for RTL values across different tissues
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted. Must include 'Tissue' and 'RTL
        columns.
    population_name : str
        The name of the population being analyzed, used in the plot title.
    annotate : bool, optional
        If True, statistical annotations will be added to the plot. Default is True.

    Returns
    -------
    None
        The function displays the boxplot but does not return any value.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.stripplot(
        x='Tissue', y='RTL', data=df,
        hue='Tissue', legend=False,
        palette=custom_palette,
        jitter=True, size=6, edgecolor='black', linewidth=0.5, ax=ax
    )

    sns.lineplot(
        x='Tissue', y='RTL', data=df,
        units='Donor',
        estimator=None,
        color='black',
        alpha=0.5,
        linewidth=1,
        ax=ax
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

    if annotate:
        pairs = list(combinations(df.Tissue.unique(), 2))

        annotator = Annotator(ax, pairs, data=df,
                              x='Tissue', y='RTL', order=df.Tissue.unique())
        annotator.configure(
            test='Mann-Whitney',
            text_format='star',
            loc='inside',
            comparisons_correction='bonferroni',
            line_width=1.2,
            hide_non_significant=True
        )
        annotator.apply_and_annotate()

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.title(f'{population_name} RTL by Tissue', fontsize=14)
    sns.despine()

    plt.tight_layout()
    plt.show()


def population_boxplot(df, annotate=True):
    """
    Creates a boxplot with overlaid stripplot and pointplot for RTL values across different populations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted. Must include 'Population' and 'RTL' columns.
    annotate : bool, optional
        If True, statistical annotations will be added to the plot. Default is True.

    Returns
    -------
    None
        The function displays the boxplot but does not return any value.
    """
    df_plot = df.reset_index(drop=True)

    pop_order = df_plot.groupby('Population')[
        'RTL'].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(16, 12))

    sns.stripplot(
        x='Population', y='RTL', data=df_plot, order=pop_order,
        hue='Population', legend=False,
        jitter=True, size=5, edgecolor='black', linewidth=0.5, ax=ax
    )
    sns.lineplot(
        x='Population', y='RTL', data=df_plot,
        units='Donor',
        estimator=None,
        color='black',
        alpha=0.5,
        linewidth=1,
        ax=ax
    )

    sns.boxplot(
        x='Population', y='RTL', data=df_plot, order=pop_order,
        hue='Population', dodge=False, boxprops=dict(alpha=0.5), ax=ax
    )

    sns.pointplot(
        x='Population', y='RTL', data=df_plot, order=pop_order, color='black',
        errorbar='sd', linestyle='none', markers='_',
        err_kws={'linewidth': 1.2}, markersize=12, linewidth=2.0, ax=ax
    )

    if annotate:
        pairs = list(combinations(pop_order, 2))

        annotator = Annotator(ax, pairs, data=df_plot,
                              x='Population', y='RTL', order=pop_order)
        annotator.configure(
            test='Mann-Whitney',
            text_format='star',
            loc='inside',
            comparisons_correction='bonferroni',
            line_width=1.2,
            hide_non_significant=True
        )
        annotator.apply_and_annotate()

    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("RTL Comparison Across Populations", fontsize=14, pad=15)
    sns.despine()

    plt.tight_layout()
    plt.show()
