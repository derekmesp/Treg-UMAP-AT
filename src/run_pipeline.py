import numpy as np
import pandas as pd
import scanpy as sc
import logging
import harmonypy as hm
import matplotlib.pyplot as plt


def clustering_pipeline(adata, sample_size=None, ARCSINH_COFACTOR=150, max_value=10, max_iter_harmony=10, theta=0, resolution=0.5, vmin=0, vmax=3):
    """
    Perform PCA, Harmony batch correction, UMAP embedding, Leiden clustering, and generates various plots for the given AnnData object.

    Parameters:
    ----------
    adata : anndata.AnnData
        AnnData object containing the data to be processed.
    sample_size : int, optional
        Number of samples to randomly select from each tissue. If None, uses the minimum tissue size.
    ARCSINH_COFACTOR : float, optional
        Coefficient for arcsinh transformation. Default is 150.
    max_value : float, optional
        Maximum value for scaling. Default is 10.
    max_iter_harmony : int, optional
        Maximum number of iterations for the Harmony integration algorithm. Default is 10.
    theta : float, optional
        Diversity clustering penalty parameter for Harmony. Default is 0.
    resolution : float, optional
        Resolution parameter for Leiden clustering. Default is 0.5.
    vmin : float, optional
        Minimum value for color scaling in UMAP plots. Default is 0.
    vmax : float, optional
        Maximum value for color scaling in UMAP plots. Default is 3.

    Returns:
    -------
    adata : anndata.AnnData
        The processed AnnData object with PCA, Harmony batch correction, UMAP embedding, and Leiden clustering results added.
    """
    if sample_size is not None:
        try:
            adata_list = [
                adata[adata.obs['tissue'] == tissue].copy()[np.random.choice(
                    adata[adata.obs['tissue'] ==
                          tissue].shape[0], sample_size, replace=False
                )]
                for tissue in adata.obs['tissue'].unique()
            ]
        except ValueError as e:
            logging.error(f"Error during sampling: {e}")
            logging.info("Falling back to minimum tissue size for sampling.")
            adata_list = [
                adata[adata.obs['tissue'] == tissue].copy()[np.random.choice(
                    adata[adata.obs['tissue'] ==
                          tissue].shape[0], adata.obs['tissue'].value_counts().min(), replace=False
                )]
                for tissue in adata.obs['tissue'].unique()
            ]
    else:
        adata_list = [
            adata[adata.obs['tissue'] == tissue].copy()[np.random.choice(
                adata[adata.obs['tissue'] ==
                      tissue].shape[0], adata.obs['tissue'].value_counts().min(), replace=False
            )]
            for tissue in adata.obs['tissue'].unique()
        ]

    for i, tissue in enumerate(adata.obs['tissue'].unique()):
        adata_list[i].obs['tissue'] = tissue

    logging.info(
        f"Concatenating {len(adata_list)} tissue-specific AnnData objects.")
    adata = sc.AnnData.concatenate(*adata_list, index_unique=None)
    adata.X = np.arcsinh(adata.X / ARCSINH_COFACTOR)
    sc.pp.scale(adata, max_value=max_value)

    logging.info("Performing PCA on the concatenated AnnData object.")
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pl.pca_variance_ratio(adata, log=False)
    sc.pl.pca_loadings(adata, components=[
                       1, 2], show=False, save=f"pca_loadings.png")

    logging.info("Running Harmony for batch correction.")
    try:
        harmony_out = hm.run_harmony(
            adata.obsm['X_pca'], adata.obs, 'sample_id', max_iter_harmony=max_iter_harmony, theta=theta)
        adata.obsm['X_pca_harmony'] = harmony_out.Z_corr
    except Exception as e:
        logging.error(f"Error during Harmony batch correction: {e}")

    sc.pp.neighbors(adata, use_rep='X_pca_harmony')
    sc.tl.umap(adata)
    logging.info("UMAP embedding completed.")

    sc.pl.umap(adata, color=['group'], cmap='turbo',
               show=False, save=f"umap_group.png")
    markers = list(adata.var_names)

    sc.tl.leiden(adata, resolution=resolution)
    logging.info("Leiden clustering completed.")

    sc.pl.umap(adata, color=['leiden'],
               title="Leiden Clustering", show=False, save=f"umap_leiden.png")
    sc.pl.umap(adata, color=['tissue'],
               title="UMAP by Tissue", show=False, save=f"umap_tissue.png")
    sc.pl.umap(adata, color=['sample_id'],
               title="UMAP by Sample ID", show=False, save=f"umap_sample_id.png")
    sc.pl.umap(adata, color=markers, cmap='turbo', vmin=vmin,
               vmax=vmax, show=False, save=f"umap_markers.png")

    for tissue in list(adata.obs.tissue.unique()):
        adata_group = adata[adata.obs['tissue'] == tissue]

        sc.pl.umap(
            adata_group,
            color=['leiden'],
            title=f'{tissue} Leiden',
            cmap='turbo',
            show=False
        )

        ax = plt.gca()
        for cluster in adata_group.obs['leiden'].cat.categories:
            cluster_mask = adata_group.obs['leiden'] == cluster
            cluster_coords = adata_group.obsm['X_umap'][cluster_mask]
            x, y = cluster_coords[:, 0].mean(), cluster_coords[:, 1].mean()
            ax.text(x, y, cluster, color='black', fontsize=10,
                    weight='bold', ha='center', va='center')
    plt.savefig(f"umap_leiden_{tissue}.png")
    plt.close()

    sc.tl.dendrogram(adata, groupby='leiden')
    sc.pl.dotplot(adata, markers, swap_axes=True, groupby='leiden',
                  cmap='RdBu_r', dendrogram=True, vcenter=0, vmin=-4, vmax=4, show=False, save=f"dotplot_leiden.png")

    marker_sets = {
        'CD4': ['CD4'],
        'CD8': ['CD8'],
        'TRM': ['CD103', 'CD69'],
        'Memory': ['CD45RA', 'CCR7']
    }
    sc.pl.dotplot(adata, marker_sets, swap_axes=True, groupby='leiden',
                  cmap='RdBu_r', dendrogram=True, vcenter=0, vmin=-4, vmax=4, show=False, save=f"dotplot_marker_sets.png")

    return adata


def dem_ranked(adata, groups='leiden', method='t-test'):
    """
    Performs differential expression analysis using the specified method and generates a dot plot of the top 3
    differentially expressed markers for each group.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data to be analyzed.
    groups : str, optional
        The name of the observation column in adata.obs that defines the groups for differential expression analysis. Default is 'leiden'.
    method : str, optional
        The method to use for differential expression analysis. Default is 't-test'.

    Returns
    -------
    """
    sc.tl.rank_genes_groups(adata, groups=groups, method=method)
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    celltype = {'celltype': []}
    cluster_to_markers = {}
    for group in groups:
        top_markers = result['names'][group][:3]
        cluster_to_markers[group] = f"{':'.join(top_markers)} ({group})"

    celltype['celltype'] = [cluster_to_markers[leiden]
                            for leiden in adata.obs['leiden']]
    cell_type_series = pd.Series(celltype['celltype'])
    unique_values = cell_type_series.unique()
    logging.info(f"Unique cell types identified: {unique_values}")

    sc.pl.rank_genes_groups_dotplot(
        adata, n_genes=3,  cmap='RdBu_r', vcenter=0, vmin=-3, vmax=3, show=False, save=f"rank_genes_groups_dotplot.png")

    return adata, unique_values
