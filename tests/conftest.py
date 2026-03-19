"""Shared fixtures for mitochontrol tests."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse


@pytest.fixture()
def small_adata():
    """Return a minimal AnnData with raw integer counts.

    60 cells, 40 genes (including 3 MT- genes and 2 RPL-
    genes).  Two rough clusters sit in different PCA
    quadrants so that Leiden can find at least 2 groups.
    """
    rng = np.random.default_rng(42)
    n_cells = 60
    n_genes = 40
    n_mt = 3
    n_ribo = 2

    gene_names = [f"Gene{i}" for i in range(n_genes - n_mt - n_ribo)]
    gene_names += [f"MT-G{i}" for i in range(n_mt)]
    gene_names += [f"RPL{i}" for i in range(n_ribo)]
    cell_names = [f"Cell{i}" for i in range(n_cells)]

    counts = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    # inject a cluster signal: first half has high expression of
    # first 10 genes, second half has high expression of next 10
    counts[:30, :10] += rng.poisson(20, size=(30, 10))
    counts[30:, 10:20] += rng.poisson(20, size=(30, 10))

    adata = AnnData(
        X=counts,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )
    adata.layers["raw_counts"] = adata.X.copy()
    return adata


@pytest.fixture()
def small_adata_sparse(small_adata):
    """Same as *small_adata* but with a sparse X matrix."""
    adata = small_adata.copy()
    adata.X = sparse.csr_matrix(adata.X)
    adata.layers["raw_counts"] = sparse.csr_matrix(
        adata.layers["raw_counts"]
    )
    return adata


@pytest.fixture()
def clustered_adata(small_adata):
    """Return a small AnnData that has already been clustered.

    Adds ``obs["leiden"]``, ``obsm["X_umap"]``, a ``lognorm``
    layer, and ``obs["mt_frac"]``.
    """
    adata = small_adata.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    adata.X = adata.layers["raw_counts"].copy()
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"]
        if "mt" in adata.var.columns
        else [],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    mt_counts = np.asarray(
        adata[:, adata.var["mt"]].X.sum(axis=1)
    ).ravel()
    total = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["mt_frac"] = np.where(
        total > 0, 100.0 * mt_counts / total, 0.0,
    )

    # minimal Leiden labels + UMAP coords
    rng = np.random.default_rng(99)
    labels = ["0"] * 30 + ["1"] * 30
    adata.obs["leiden"] = pd.Categorical(labels)
    adata.obsm["X_umap"] = rng.standard_normal((60, 2))

    return adata
