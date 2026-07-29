import matplotlib.pyplot as plt
import scanpy as sc


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
