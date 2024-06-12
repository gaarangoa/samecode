import pandas as pd
import numpy as np

class TPMNormalizer:
    def __init__(self, gene_lengths_df):
        """
        Initializes the TPMNormalizer class.
        
        Parameters:
        - gene_lengths_df (pandas.DataFrame): DataFrame with columns ['gene', 'length'].
        """
        self.gene_lengths = gene_lengths_df.set_index('gene')['length']

    def normalize(self, count_df):
        """
        Normalize raw count data using TPM.
        
        Parameters:
        - count_df (pandas.DataFrame): DataFrame with rows as samples and columns as genes,
                                       where entries are raw count data.
                                       
        Returns:
        - pandas.DataFrame: DataFrame of TPM normalized data.
        
        # # Example usage:
        # gene_lengths_df = pd.DataFrame({'gene': ['Gene1', 'Gene2'], 'length': [1, 1]})
        # count_df = pd.DataFrame({
        #     'Gene1': [1],
        #     'Gene2': [2],
        # }, index=['Sample1'])

        # tpm_normalizer = TPMNormalizer(gene_lengths_df)
        # normalized_counts = tpm_normalizer.normalize(count_df)
        # print(normalized_counts)
        
        """
        # Ensure the gene order matches and filter to genes with known lengths
        genes_in_both = self.gene_lengths.index.intersection(count_df.columns)
        gene_lengths = self.gene_lengths[genes_in_both].sort_index()
        count_df = count_df[genes_in_both].sort_index(axis=1)
        
        # Convert gene lengths from bases to kilobases for TPM calculation
        gene_lengths_kb = gene_lengths / 1e3
        
        # Calculate scaled counts (count divided by gene length in kb)
        scaled_counts = count_df.div(gene_lengths_kb)
        
        # Sum scaled counts across all genes for each sample to get the scaling factor
        scaling_factor = scaled_counts.sum(axis=1)
        
        # Divide scaled counts by the scaling factor and multiply by 1,000,000 for TPM
        tpm = (scaled_counts.T / scaling_factor).T * 1e6
        
        return tpm


