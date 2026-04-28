import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Log2Transformer(BaseEstimator, TransformerMixin):
    """log2(X + offset) transformer. Preserves DataFrame columns/index.

    Parameters
    ----------
    offset : float, default=1
        Added before log2. Common choices: 1 for log2(TPM+1),
        0.001 for log2(TPM+0.001).
    """

    def __init__(self, offset=1):
        self.offset = offset

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        vals = X.values if isinstance(X, pd.DataFrame) else X
        if (vals[~np.isnan(vals)] < 0).any():
            raise ValueError("Negative values found in input. TPM data cannot contain negatives.")
        clean = X.fillna(0) if isinstance(X, pd.DataFrame) else np.nan_to_num(X, nan=0.0)
        result = np.log2(clean + self.offset)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, index=X.index, columns=X.columns)
        return result

    def inverse_transform(self, X):
        result = np.power(2, X) - self.offset
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, index=X.index, columns=X.columns)
        return result


class CombatBatchCorrector(BaseEstimator, TransformerMixin):
    """Sklearn-compatible ComBat batch effect correction.

    Expects already-transformed data (e.g. log2(TPM+1)) as input.
    Use Log2Transformer upstream or a Pipeline to compose them.

    Pipeline:
    1. Filter low-variance genes (var < min_variance)
    2. PCA (before)
    3. ComBat batch correction
    4. PCA (after)

    Parameters
    ----------
    min_variance : float, default=0.01
        Genes with variance below this threshold are dropped
        before ComBat.
    """

    def __init__(self, min_variance=0.01):
        self.min_variance = min_variance

    def fit(self, X, y=None, **fit_params):
        """No-op. ComBat is transductive — use fit_transform() instead."""
        return self

    def transform(self, X, y=None):
        """Not supported. ComBat is transductive and cannot apply a
        pre-learned correction to new data. Use fit_transform()."""
        raise NotImplementedError(
            "ComBat is transductive — it estimates and removes batch effects "
            "in a single pass over all samples. Call fit_transform(X, batch) "
            "instead."
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Run the full batch correction pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Expression matrix (samples x genes), already log-transformed.
        y : array-like of shape (n_samples,)
            Batch label for each sample. Uses sklearn's y slot so that
            labels propagate through Pipeline.fit_transform(X, y).

        Returns
        -------
        corrected : pd.DataFrame
            Batch-corrected expression matrix (same scale as input).
            Columns are the subset of genes that survived variance filtering.
        """
        from combat.pycombat import pycombat

        batch = np.asarray(y)
        expr = X.copy()
        self.n_genes_input_ = expr.shape[1]

        # --- variance filter ---
        gene_vars = expr.var(axis=0)
        high_var = gene_vars[gene_vars > self.min_variance].index
        expr = expr[high_var]

        self.genes_kept_ = list(expr.columns)
        self.n_genes_kept_ = len(self.genes_kept_)

        # --- PCA before correction ---
        scaled_before = StandardScaler().fit_transform(expr.fillna(0))
        n_components = min(2, expr.shape[1], expr.shape[0])
        pca_before = PCA(n_components=n_components)
        pcs_before = pca_before.fit_transform(scaled_before)
        self.pca_before_ = pca_before
        self.pcs_before_ = pcs_before

        # --- ComBat (expects genes x samples, no NaN allowed) ---
        expr = expr.fillna(0)
        corrected_combat = pycombat(expr.T, batch)
        corrected = pd.DataFrame(
            corrected_combat.T.values,
            index=X.index,
            columns=expr.columns,
        )

        # --- PCA after correction ---
        scaled_after = StandardScaler().fit_transform(corrected.fillna(0))
        pca_after = PCA(n_components=n_components)
        pcs_after = pca_after.fit_transform(scaled_after)
        self.pca_after_ = pca_after
        self.pcs_after_ = pcs_after
        self.batch_labels_ = batch

        self.corrected_ = corrected
        return corrected

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def plot_pca(self, batch=None, title='', figsize=(16, 6)):
        """Plot before/after PCA colored by batch."""
        batch = batch if batch is not None else self.batch_labels_
        unique_batches = sorted(set(batch))

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, pcs, pca_obj, label in [
            (axes[0], self.pcs_before_, self.pca_before_, 'BEFORE correction'),
            (axes[1], self.pcs_after_, self.pca_after_, 'AFTER correction'),
        ]:
            for b in unique_batches:
                mask = np.array(batch) == b
                ax.scatter(pcs[mask, 0], pcs[mask, 1], label=b, alpha=0.7, s=40)
            ax.set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title(f'{label}')
            ax.legend(fontsize=8)

        fig.suptitle(f'{title} - ComBat batch correction', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Return a dict of correction diagnostics."""
        from scipy.spatial.distance import pdist, squareform

        def _centroid_dist(pcs, batch):
            df = pd.DataFrame({'PC1': pcs[:, 0], 'PC2': pcs[:, 1], 'batch': batch})
            centroids = df.groupby('batch')[['PC1', 'PC2']].mean().values
            if len(centroids) < 2:
                return 0.0
            return squareform(pdist(centroids, 'euclidean')).mean()

        before_dist = _centroid_dist(self.pcs_before_, self.batch_labels_)
        after_dist = _centroid_dist(self.pcs_after_, self.batch_labels_)
        reduction = ((before_dist - after_dist) / before_dist * 100) if before_dist > 0 else 0

        return {
            'n_genes_input': self.n_genes_input_,
            'n_genes_kept': self.n_genes_kept_,
            'n_genes_removed': self.n_genes_input_ - self.n_genes_kept_,
            'batch_centroid_dist_before': round(before_dist, 3),
            'batch_centroid_dist_after': round(after_dist, 3),
            'batch_separation_reduction_pct': round(reduction, 1),
            'pc1_var_before': round(self.pca_before_.explained_variance_ratio_[0] * 100, 2),
            'pc1_var_after': round(self.pca_after_.explained_variance_ratio_[0] * 100, 2),
        }

class GeneMapperDedup(BaseEstimator, TransformerMixin):
    """Filter columns to reference gene set (preserving ref order) and
    deduplicate by subject ID.

    Parameters
    ----------
    ref_genes : list
        Reference gene list. Output columns follow this ordering.
    subject_col : str, default='USUBJID'
        Column used for deduplication.
    extra_cols : list or None, default=None
        Additional columns to keep (e.g. ['batch', 'pipeline']).
    keep : str or False, default='first'
        Which duplicate to keep (passed to drop_duplicates).
        Use False to drop all duplicates.
    """

    def __init__(self, ref_genes, subject_col='USUBJID', extra_cols=None, keep='first'):
        self.ref_genes = ref_genes
        self.subject_col = subject_col
        self.extra_cols = extra_cols or []
        self.keep = keep

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        available = set(X.columns)
        mapped_genes = [g for g in self.ref_genes if g in available]
        self.mapped_genes_ = mapped_genes

        keep_cols = mapped_genes + [self.subject_col]
        keep_cols += [c for c in self.extra_cols if c in available]

        out = X[keep_cols].copy()

        n_before = out.shape[0]
        out = out.drop_duplicates(self.subject_col, keep=self.keep)
        n_after = out.shape[0]

        self.n_genes_mapped_ = len(mapped_genes)
        self.n_samples_before_ = n_before
        self.n_samples_after_ = n_after
        self.n_duplicates_removed_ = n_before - n_after

        print(f'Gene mapping: {X.shape[1]} cols → {len(mapped_genes)} genes')
        print(f'Dedup: {n_before} → {n_after} samples (removed {n_before - n_after})')

        return out
