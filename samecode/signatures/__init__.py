import pandas as pd

def median_aggregation(data, signatures={}):
    '''

    **Summary**: Computes the median of a list of gene expressions for each sample in a dataset.

    **Arguments**:

    * `data`: A pandas DataFrame containing the gene expression data.
    * `signatures`: A dictionary of gene signatures, where each key is a sample name and each value is a list of genes to be aggregated.

    **Returns**: A new pandas DataFrame with the median of each gene expression for each sample. The columns are the sample names, and the rows are the genes.

    **Description**:

    The `median_aggregation` function takes a dataset of gene expression data and a dictionary of gene signatures, where each key is a sample name and each value is a list of genes to be aggregated. It computes the median of the gene expressions for each sample and returns a new DataFrame with the results.

    The function first loops through each signature in the `signatures` dictionary and computes the median of the gene expressions for each sample. The resulting values are then appended to a list of lists, where each sublist contains the median values for one sample. Finally, the function creates a new DataFrame from the list of lists and sets the columns to the sample names.

    **Examples**:

    Here is an example usage of the `median_aggregation` function:
    ```
    # Load the dataset
    data = pd.read_csv('gene_expression_data.csv')

    # Define the gene signatures
    signatures = {'signature1': ['gene1', 'gene2'], 
                'signature2': ['gene3', 'gene4'], 
                'signature3': ['gene5']}

    # Apply the median aggregation
    df = median_aggregation(data, signatures)

    # Print the results
    print(df)
    ```
    In this example, the `median_aggregation` function takes a dataset of gene expression data and a dictionary of gene signatures. It computes the median of the gene expressions for each sample and returns a new DataFrame with the results. The resulting DataFrame has three columns, each corresponding to one of the samples, and four rows, each corresponding to one of the genes.
    
    '''
    values = []
    names = []
    for name, genes in signatures.items():
        
        value = data[genes].median(axis=1)
        values.append(value)
        names.append(name)
    
    df = pd.DataFrame(values).T
    df.columns = names
    return df