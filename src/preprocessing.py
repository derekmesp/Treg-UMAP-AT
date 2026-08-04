import logging

import flowkit as fk
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import re

logging.basicConfig(level=logging.INFO)


def read_flow(wsp_directory, fcs_directory, gate):
    """
    Reads flow cytometry data from FCS files in specified directory.

    This function loads This function loads flow cytometry data from FCS files, processes them into a pandas DataFrame,
    and assigns tissue and condition labels based on sample IDs. It automatically detects conditions from the sample names.

    Parameters:
    -----------
    wsp_directory : str
        Path to the wsp file containing gating for FCS files.
    fcs_directory : str
        Path to the directory containing FCS files to be analyzed.
    gate : str
        Name of gate containing population of interest.

    Returns:
    --------
    tuple
        A tuple containing three elements:
        - df_flow (pandas.DataFrame): Combined DataFrame of all flow cytometry samples with added
          metadata columns (tissue, sample_id, condition).
        - sample_list (list): List of all sample IDs found in the directory.
        - wsp (flowkit.Workspace): The flowkit Workspace object containing gating and event information.

    """

    wsp = fk.Workspace(wsp_directory, fcs_directory)
    wsp.analyze_samples(group_name='All Samples', verbose=False, use_mp=False)
    sample_list = wsp.get_sample_ids(group_name='All Samples')

    df_flow = []
    for sample_id in sample_list:
        df = wsp.get_gate_events(sample_id, gate)
        df['sample_id'] = (sample_id.split(' ')[1])
        df['tissue'] = (sample_id.split(' ')[2])
        df['condition'] = (sample_id.split(' ')[3])
        df_flow.append(df)

    df_flow = pd.concat(df_flow)

    new_cols = []
    for col in df_flow.columns:
        if ' : ' in col:
            marker = re.search(r'-A (.*?) :', col)
            new_cols.append(marker.group(1) if marker else col)
        else:
            new_cols.append(col)

    df_flow.columns = new_cols
    print('Parameters:', df_flow.keys())
    return df_flow, sample_list, wsp


def pd_to_adata(df_flow, df_flow_counts):
    """
    Converts flow cytometry data from pandas DataFrames to an AnnData object.

    This function processes flow cytometry data and associated counts, creating an AnnData object
    with appropriate metadata. It truncates sample IDs, creates a metadata DataFrame, and assigns
    group and sample ID information to the AnnData object's observation annotations.

    Parameters:
    -----------
    df_flow : pandas.DataFrame
        A DataFrame containing flow cytometry data, including 'sample_id', 'condition', and 'tissue' columns.
    df_flow_counts : pandas.DataFrame
        A DataFrame containing count data for the flow cytometry samples.

    Returns:
    --------
    anndata.AnnData
        An AnnData object containing the flow cytometry count data with associated metadata.
        The object includes:
        - X: The count matrix from df_flow_counts
        - obs: Observation annotations including 'group' and 'sample_id'

    """

    if df_flow.isna().any().any():
        nan_cols = df_flow.columns[df_flow.isna().any()].tolist()
        raise ValueError(
            f"NaN values detected in df_flow columns: {nan_cols}"
        )

    df_flow['sample_id'] = df_flow['sample_id'].apply(lambda x: x[:4])
    list_metadata = {
        'group': df_flow.condition,
        'sample_id': df_flow.sample_id,
        'tissue': df_flow.tissue
    }

    df_metadata = pd.DataFrame(list_metadata)
    adata = sc.AnnData(df_flow_counts)
    df_metadata.index = adata.obs.index

    for col in df_metadata.columns:
        adata.obs[col] = df_metadata[col]

    return adata


def RTL_cleanup(dir):
    """
    Cleans and preprocesses the RTL data from a CSV file.

    Parameters
    ----------
    dir : str
        The directory path to the CSV file containing the RTL data.

    Returns
    -------
    tuple
        A tuple containing:
        - df (pandas.DataFrame): The cleaned DataFrame with RTL data.
        - pop_dfs (dict): A dictionary of DataFrames, each corresponding to a unique population in the data.
    """
    df = pd.read_excel(dir, index_col=0)
    df['RTL'] = pd.to_numeric(df['RTL'], errors='coerce')

    try:
        df['Donor'] = df.index
    except Exception as e:
        logging.error(f"Error occurred while setting Donor column: {e}")

    try:
        df['Age group'] = np.where(df['Age'] < 18, 'Non-adult', 'Adult')
        logging.info("Age group column added successfully.")
    except Exception as e:
        logging.error(f"Error occurred while adding Age group column: {e}")

    try:
        df['Tissue residence'] = np.where(
            df['Population'].str.contains(
                r'CD103\+CD69\+', regex=True, na=False),
            'TRM',
            'Non-TRM'
        )
        logging.info("Tissue residence column added successfully.")

    except Exception as e:
        logging.error(
            f"Error occurred while adding Tissue residence column: {e}")

    df = df.dropna(subset=['RTL']).query('RTL != 0')
    logging.info(
        f"DataFrame after dropping NaN and zero RTL values: {df.shape[0]} rows remaining.")

    df = df.query('RTL >= 0')
    logging.info(
        f"DataFrame after converting RTL to numeric and filtering negative values: {df.shape[0]} rows remaining.")

    df = df.reset_index(drop=True)
    pop_dfs = {pop: df[df.Population == pop]
               for pop in df.Population.unique()}
    logging.info(
        f"DataFrames created for each population: {', '.join(pop_dfs.keys())}")

    return df, pop_dfs
