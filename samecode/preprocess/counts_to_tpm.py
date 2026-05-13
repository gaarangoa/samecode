"""
Sklearn-compatible pipeline for converting raw gene counts to TPM.

Usage:
    from counts_to_tpm import TPMPipeline

    pipe = TPMPipeline(reference="gencode.v44.annotation.gtf", id_attr="gene_name", log2=True)
    tpm_df = pipe.fit_transform(counts_df)

Or step by step:
    from counts_to_tpm import ReferenceGenes, TPMTransformer, Log2Transformer
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("tpm", TPMTransformer(reference="annotation.gtf", id_attr="gene_id")),
        ("log2", Log2Transformer()),
    ])
    tpm_df = pipe.fit_transform(counts_df)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ReferenceGenes:
    """Parse GTF/GFF3 to compute effective gene lengths (merged exon spans)."""

    def __init__(self, path: str, id_attr: str = "gene_id"):
        self.path = path
        self.id_attr = id_attr
        self._lengths = None

    @property
    def lengths(self) -> pd.Series:
        if self._lengths is None:
            self._lengths = self._parse()
        return self._lengths

    def _detect_format(self) -> str:
        suffix = Path(self.path).suffix.lower()
        if suffix in (".gff3", ".gff"):
            return "gff3"
        return "gtf"

    def _parse(self) -> pd.Series:
        fmt = self._detect_format()
        exons: dict[str, list[tuple[str, int, int]]] = {}

        with open(self.path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                feature_type = parts[2]
                if feature_type != "exon":
                    continue

                chrom = parts[0]
                start = int(parts[3])
                end = int(parts[4])
                attrs = parts[8]

                gene_id = self._extract_attr(attrs, fmt)
                if gene_id is None:
                    continue

                exons.setdefault(gene_id, []).append((chrom, start, end))

        lengths = {}
        for gene_id, intervals in exons.items():
            lengths[gene_id] = self._merged_length(intervals)

        series = pd.Series(lengths, dtype=np.float64)
        series.index.name = "gene_id"
        return series[series > 0]

    def _extract_attr(self, attrs: str, fmt: str) -> str | None:
        if fmt == "gff3":
            for field in attrs.split(";"):
                if "=" not in field:
                    continue
                key, val = field.split("=", 1)
                if key.strip().lower() == self.id_attr.lower():
                    return val.strip().split(",")[0]
                if self.id_attr == "gene_name" and key.strip() == "Name":
                    return val.strip()
            # GFF3: try Parent for exons -> look for gene_id in parent chain
            return None
        else:
            for field in attrs.split(";"):
                field = field.strip()
                if not field:
                    continue
                parts = field.split(None, 1)
                if len(parts) == 2 and parts[0] == self.id_attr:
                    return parts[1].strip('"').split(".")[0] if self.id_attr == "gene_id" else parts[1].strip('"')
            return None

    @staticmethod
    def _merged_length(intervals: list[tuple[str, int, int]]) -> int:
        by_chrom: dict[str, list[tuple[int, int]]] = {}
        for chrom, start, end in intervals:
            by_chrom.setdefault(chrom, []).append((start, end))

        total = 0
        for chrom, spans in by_chrom.items():
            spans.sort()
            merged_start, merged_end = spans[0]
            for s, e in spans[1:]:
                if s <= merged_end:
                    merged_end = max(merged_end, e)
                else:
                    total += merged_end - merged_start + 1
                    merged_start, merged_end = s, e
            total += merged_end - merged_start + 1
        return total


class TPMTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer: raw counts -> TPM using a reference annotation."""

    def __init__(self, reference: str = None, id_attr: str = "gene_id"):
        self.reference = reference
        self.id_attr = id_attr

    def fit(self, X, y=None):
        if self.reference is None:
            raise ValueError("reference annotation file path is required")
        self._ref = ReferenceGenes(self.reference, id_attr=self.id_attr)
        self.gene_lengths_ = self._ref.lengths
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            genes = X.index
            columns = X.columns
        else:
            X = pd.DataFrame(X)
            genes = X.index
            columns = X.columns

        common_genes = genes.intersection(self.gene_lengths_.index)
        dropped = len(genes) - len(common_genes)
        if dropped > 0:
            warnings.warn(f"{dropped} genes have no matching length in reference and were dropped.")

        counts = X.loc[common_genes] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, index=genes, columns=columns).loc[common_genes]
        gene_lengths_kb = self.gene_lengths_.loc[common_genes] / 1000.0

        rate = counts.div(gene_lengths_kb, axis=0)
        tpm = rate.div(rate.sum(axis=0), axis=1) * 1e6

        return tpm

    def get_feature_names_out(self, input_features=None):
        return input_features


class Log2Transformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer: apply log2(X + 1)."""

    def __init__(self, pseudocount: float = 1.0):
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log2(X + self.pseudocount)

    def get_feature_names_out(self, input_features=None):
        return input_features


def TPMPipeline(reference: str, id_attr: str = "gene_id", log2: bool = False) -> Pipeline:
    """Convenience factory: builds a ready-to-use sklearn Pipeline."""
    steps = [("tpm", TPMTransformer(reference=reference, id_attr=id_attr))]
    if log2:
        steps.append(("log2", Log2Transformer()))
    return Pipeline(steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert raw gene counts to TPM using a reference annotation.")
    parser.add_argument("--counts", required=True, help="Path to counts matrix (CSV/TSV). Genes as rows, samples as columns.")
    parser.add_argument("--reference", required=True, help="Path to GTF/GFF3 annotation file.")
    parser.add_argument("--id-attr", default="gene_id", help="Attribute to match genes on (gene_id or gene_name).")
    parser.add_argument("--output", required=True, help="Output path for TPM matrix.")
    parser.add_argument("--log2", action="store_true", help="Apply log2(TPM + 1) transformation.")
    args = parser.parse_args()

    ext = args.counts.rsplit(".", 1)[-1].lower()
    sep = "\t" if ext in ("tsv", "txt") else ","
    counts = pd.read_csv(args.counts, sep=sep, index_col=0)

    pipe = TPMPipeline(reference=args.reference, id_attr=args.id_attr, log2=args.log2)
    tpm = pipe.fit_transform(counts)

    out_ext = args.output.rsplit(".", 1)[-1].lower()
    out_sep = "\t" if out_ext in ("tsv", "txt") else ","
    tpm.to_csv(args.output, sep=out_sep)
    print(f"TPM matrix written to {args.output} ({tpm.shape[0]} genes x {tpm.shape[1]} samples)")
