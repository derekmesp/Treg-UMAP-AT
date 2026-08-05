from itertools import combinations
import logging

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from statannotations.Annotator import Annotator

logging.basicConfig(level=logging.INFO)

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
    Creates a boxplot with overlaid stripplot and pointplot for RTL values across different tissue
    types within a specified population.

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

    df = df.reset_index(drop=True)
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

        tissue_counts = df['Tissue'].value_counts()
        valid_tissues = tissue_counts[tissue_counts >= 2].index.tolist()

        if len(valid_tissues) >= 2:
            pairs = list(combinations(valid_tissues, 2))

            annotator = Annotator(ax, pairs, data=df,
                                  x='Tissue', y='RTL', order=df.Tissue.unique())
            annotator.configure(
                test='Mann-Whitney',
                text_format='star',
                loc='inside',
                comparisons_correction='bonferroni',
                line_width=1.2,
                hide_non_significant=True,
                verbose=False
            )

            try:
                annotator.apply_and_annotate()
            except Exception as e:
                logging.warning(
                    f"Skipping statistical annotations for {population_name} due to an internal error: {e}")
        else:
            print(
                f"Skipping statistical annotation for {population_name}: Insufficient sample counts across tissue groups.")

    outlier_donors = df[df.groupby('Population')['RTL'].transform(
        lambda x: (x - x.mean()).abs() > 2 * x.std())]['Donor'].unique()
    logging.info(
        f"Outlier donors detected: {', '.join(str(d) for d in outlier_donors)}")

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.title(f'{population_name} RTL by Tissue', fontsize=14)
    sns.despine()

    plt.tight_layout()
    plt.show()


def population_boxplot(df, split, annotate=True, subset=None, plot_donors=False):
    """
    Creates a boxplot with overlaid stripplot and pointplot for RTL values across different populations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted. Must include 'Population' and 'RTL' columns.
    split : str
        The name of the column in df to split the data by (e.g., 'Population', 'Age group').
    annotate : bool, optional
        If True, statistical annotations will be added to the plot. Default is True.
    subset : list, optional
        A list of specific values in the split column to include in the plot. If None, all values are included.
    plot_donors : bool, optional
        If True, donor IDs will be annotated next to their corresponding data points. Default is False.

    Returns
    -------
    None
        The function displays the boxplot but does not return any value.
    """

    df_plot = df.reset_index(drop=True)

    if subset is not None:
        matching_cols = [
            col for col in df_plot.columns
            if set(subset).issubset(set(df_plot[col].dropna().unique()))
        ]

        if matching_cols:
            matched_column = matching_cols[0]
            df_plot = df_plot[df_plot[matched_column].isin(subset)]
        else:
            logging.error(
                f"None of the DataFrame columns contain values from the subset: {subset}")

    pop_order = df_plot.groupby(split)[
        'RTL'].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(16, 12))

    sns.stripplot(
        x=split, y='RTL', data=df_plot, order=pop_order,
        hue=split, legend=False,
        jitter=True, size=5, edgecolor='black', linewidth=0.5, ax=ax
    )
    sns.lineplot(
        x=split, y='RTL', data=df_plot,
        units='Donor',
        estimator=None,
        color='black',
        alpha=0.5,
        linewidth=1,
        ax=ax
    )

    sns.boxplot(
        x=split, y='RTL', data=df_plot, order=pop_order,
        hue=split, dodge=False, boxprops=dict(alpha=0.5), ax=ax
    )

    sns.pointplot(
        x=split, y='RTL', data=df_plot, order=pop_order, color='black',
        errorbar='sd', linestyle='none', markers='_',
        err_kws={'linewidth': 1.2}, markersize=12, linewidth=2.0, ax=ax
    )

    if annotate:
        pairs = list(combinations(pop_order, 2))

        annotator = Annotator(ax, pairs, data=df_plot,
                              x=split, y='RTL', order=pop_order)
        annotator.configure(
            test='Mann-Whitney',
            text_format='star',
            loc='inside',
            comparisons_correction='bonferroni',
            line_width=1.2,
            hide_non_significant=True,
            verbose=False
        )
        _, annotations = annotator.apply_and_annotate()

        for ann in annotations:
            stat_res = ann.data
            if stat_res.pvalue is not None and stat_res.pvalue < 0.05:
                group1 = stat_res.group1
                group2 = stat_res.group2

                logging.info(
                    f"Significant difference between {group1} and {group2} "
                    f"({stat_res.test_short_name}, corrected p-val: {stat_res.pvalue:.4e})"
                )

    outlier_donors = df_plot[df_plot.groupby(split)['RTL'].transform(
        lambda x: (x - x.mean()).abs() > 2 * x.std())]['Donor'].unique()
    if len(outlier_donors) > 0:
        logging.info(
            f"Outlier donors detected: {', '.join(str(d) for d in outlier_donors)}")

    if plot_donors:
        cat_to_idx = {cat: i for i, cat in enumerate(pop_order)}
        stripplot_collections = [
            c for c in ax.collections if hasattr(c, 'get_offsets')]

        if stripplot_collections:
            all_dots = []
            for col in stripplot_collections:
                all_dots.extend(col.get_offsets())

            df_text = df_plot.copy()
            df_text['sort_cat'] = df_text[split].map(cat_to_idx)
            df_text = df_text.sort_values(
                by=['sort_cat', 'RTL']).reset_index(drop=True)

            for idx, row in df_text.iterrows():
                if idx < len(all_dots):
                    dot_x, dot_y = all_dots[idx]

                    ax.text(
                        x=dot_x + 0.03,
                        y=dot_y,
                        s=str(row["Donor"]),
                        fontsize=8,
                        color="black",
                        alpha=0.7,
                        va="center",
                        ha="left"
                    )
            logging.info(
                f"Annotated donor IDs for {len(df_text)} data points in the plot.")
        else:
            for idx, row in df_plot.iterrows():
                x_position = cat_to_idx[row[split]]
                ax.text(
                    x=x_position + 0.15,
                    y=row["RTL"],
                    s=str(row["Donor"]),
                    fontsize=8,
                    color="black",
                    alpha=0.7,
                    va="center"
                )

    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("{} RTL Comparison Across {}".format(", ".join(subset) if subset is not None else "",
                                                   split), fontsize=14, pad=15)
    sns.despine()

    plt.tight_layout()
    plt.show()


def age_RTL_plot(df, key):
    """
    Creates a scatter plot of RTL values against age.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted. Must include 'Age' and 'RTL' columns.
    key : str
        A string used in the plot title to indicate the specific subset or condition being visualized.

    Returns
    -------
    None
        The function displays the scatter plot but does not return any value.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        x='Age', y='RTL', data=df,
        palette=custom_palette, hue='Tissue', s=60, edgecolor='black', linewidth=0.5
    )

    for idx, row in df.iterrows():
        ax.text(
            x=row["Age"] + 0.2,
            y=row["RTL"],
            s=str(row["Donor"]),
            fontsize=9,
            color="black",
            alpha=0.8,
            va="center"
        )
    plt.plot(df['Age'], df['RTL'], marker='o',
             zorder=0, color='gray', alpha=0.5)

    sns.despine()
    plt.tight_layout()
    plt.grid()
    plt.title("RTL vs Age for {}".format(key), fontsize=14, pad=15)
    plt.show()
