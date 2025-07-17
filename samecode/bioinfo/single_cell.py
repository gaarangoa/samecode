import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
def pseudobulk_tpm(
    adata,
    groupby,
    length_key: str = "feature_length",
    length_key_id: str = 'ensembl_id',
    layer=None,
    minimum_cells: int = 1,
):
    adata.var_names = adata.var[length_key_id]
    gene_lengths = pd.DataFrame(adata.var[length_key])
    gene_lengths.index = adata.var[length_key_id]
    gene_lengths = gene_lengths[length_key]
    
    # Argument checks
    if groupby not in adata.obs.columns:
        raise ValueError(f"'{groupby}' not in adata.obs")
    if not adata.var_names.isin(gene_lengths.index).all():
        missing = adata.var_names[~adata.var_names.isin(gene_lengths.index)]
        raise ValueError(f"gene_lengths missing {len(missing)} genes, e.g. {missing[:5].tolist()}")
    if (gene_lengths <= 0).any():
        raise ValueError("gene_lengths must be positive")

    # Choose counts matrix
    X = adata.layers[layer] if layer else adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()            # densify sparse matrix

    # Pre-compute per-cell → group mapping
    groups = adata.obs[groupby].astype("category")
    group_codes = groups.cat.codes.values      # int array
    group_names = groups.cat.categories.tolist()
    n_groups = len(group_names)

    # Sum counts across cells within each group
    # Result shape: n_groups × n_genes
    summed = np.zeros((n_groups, adata.n_vars), dtype=np.float64)
    for g in tqdm(range(n_groups)):
        idx = np.where(group_codes == g)[0]
        if len(idx) < minimum_cells:
            continue
        summed[g, :] = X[idx, :].sum(axis=0)

    summed_df = pd.DataFrame(
        summed.T,       # genes × groups
        index=adata.var_names,
        columns=group_names,
        dtype=float
    )

    # Convert counts → RPK
    lengths_kb = gene_lengths.loc[adata.var_names].values / 1000.0
    rpk = summed_df.divide(lengths_kb, axis=0)

    # Convert RPK → TPM
    scale = rpk.sum(axis=0) / 1e6
    tpm = rpk.divide(scale, axis=1)

    return tpm
